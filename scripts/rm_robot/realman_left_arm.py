import json
import time
import logging
import socket
import numpy as np
# ROS2相关导入（替代rospy）
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

try:
    from rm_ros_interfaces.msg import Sixforce
except ImportError:
    Sixforce = None

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===================== 手动配置参数 =====================
# 机械臂网络参数
ARM_IP = "192.168.1.19"          # 机械臂IP
ARM_PORT = 8080                  # 机械臂TCP端口
LOCAL_IP = "192.168.1.20"        # 本地IP
LOCAL_PORT = 8099                # 本地UDP端口

# 功能开关
GET_VEL = True                   # 是否获取关节速度
GET_TORQUE = True                # 是否获取关节扭矩
GET_SIX_FORCE = True             # 是否在 UDP 中启用并发布末端六维力（需选配力传感器）
FORCE_COORDINATE = 0             # 与 rm_driver udp_force_coordinate 一致：0 传感器系 1 工作系 2 工具系；-1 为协议“不支持力”
SIX_FORCE_TOPIC = "/left/rm_driver/get_force_data_result"  # 与 rm_driver 同名，便于原 rosbag/工具链
# UDP 里若没有 force_sensor，可建第二条 TCP 专查 get_force_data（周期≥50ms）；单连接控制器请改为 False
SIX_FORCE_USE_TCP_QUERY = True
SIX_FORCE_TCP_MIN_PERIOD = 0.05  # 官方建议查询间隔不小于 50ms
SIX_FORCE_PREFER_ZERO = False    # True：优先用系统外受力 zero_force / zero_force_data

# 机械臂参数
ARM_AXIS = 7                     # 机械臂轴数
ARM_KI = [7, 7, 7, 3, 3, 3, 3]   # 电流转扭矩系数

# ROS2配置
ROS2_JOINT_TOPIC = "/gello/left_arm/joint_states"

# 全局变量：存储ROS2订阅的角度值（角度制）
ROS2_JOINT_ANGLES_DEG = None
ROS2_GRIPPER = None
class RmArm:
    def __init__(self):
        """初始化机械臂的网络连接"""
        self.arm_ip = ARM_IP
        self.get_vel = GET_VEL
        self.get_torque = GET_TORQUE
        self.get_six_force = GET_SIX_FORCE
        self.force_coordinate = FORCE_COORDINATE
        self.arm_port = ARM_PORT
        self.local_ip = LOCAL_IP
        self.local_port = LOCAL_PORT
        self._filt_joint_deg = None
        self._last_sent_deg = None
        self._last_filter_t = time.monotonic()
        self._force_tcp = None
        self._force_tcp_last_t = 0.0
        self._udp_packets = 0
        self._udp_force_warned = False
        # 1. 建立TCP连接到机械臂
        logging.info(f"正在连接机械臂 TCP: {self.arm_ip}:{self.arm_port}")
        self.arm = socket.socket()
        self.arm.connect((self.arm_ip, self.arm_port))
        logging.info("✅ TCP连接成功")

        # 2. 发送UDP实时推送配置
        set_udp = {
            "command":"set_realtime_push",
            "cycle":2,
            "enable":True,
            "port":self.local_port,
            "ip":self.local_ip,
            "custom":{
                "joint_speed":True,
                "arm_current_status":True
            }
        }
        if self.get_six_force and self.force_coordinate >= 0:
            set_udp["force_coordinate"] = self.force_coordinate
        self.arm.send(json.dumps(set_udp).encode('utf-8'))
        _ = self.arm.recv(1024)
        logging.info("✅ UDP实时推送配置发送成功")

        # 3. 机械臂基础参数
        self.arm_axis = ARM_AXIS
        self.arm_ki = ARM_KI

        # 4. 初始化UDP套接字
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind((self.local_ip, self.local_port))
        logging.info(f"✅ UDP绑定成功: {self.local_ip}:{self.local_port}")

        if self.get_six_force and SIX_FORCE_USE_TCP_QUERY:
            try:
                fs = socket.socket()
                fs.settimeout(3.0)
                fs.connect((self.arm_ip, self.arm_port))
                self._force_tcp = fs
                logging.info("✅ 六维力辅助 TCP 已连接（JSON get_force_data 轮询，与 CANFD 主连接分离）")
            except OSError as e:
                logging.warning("六维力辅助 TCP 未建立（仅使用 UDP 力数据）: %s", e)

    @staticmethod
    def _normalize_six(arr):
        if arr is None:
            return None
        v = np.asarray(arr, dtype=float).reshape(-1)
        if v.size < 6:
            return None
        six = v[:6].astype(float)
        if np.max(np.abs(six)) > 1.0e4:
            six *= 1.0e-3
        return six

    def _extract_six_force_from_udp(self, payload):
        """从 UDP JSON 提取 6 维力；兼容多种字段名。"""
        if not self.get_six_force:
            return None
        top_force = payload.get("force_data") or payload.get("six_force")
        if top_force is not None:
            return self._normalize_six(top_force)

        fs = None
        for key in ("force_sensor", "Force_Sensor", "forceSensor"):
            fs = payload.get(key)
            if isinstance(fs, dict):
                break
            fs = None
        if not isinstance(fs, dict):
            return None

        primary, secondary = ("zero_force", "force") if SIX_FORCE_PREFER_ZERO else ("force", "zero_force")
        for name in (primary, secondary, "Force", "Zero_force"):
            fval = fs.get(name)
            if fval is not None:
                six = self._normalize_six(fval)
                if six is not None:
                    return six
        return None

    def query_six_force_tcp(self):
        """第二条 TCP 上查询 get_force_data（不干扰主连接上的 movej_canfd）。"""
        if not self._force_tcp or not self.get_six_force:
            return None
        now = time.monotonic()
        if now - self._force_tcp_last_t < SIX_FORCE_TCP_MIN_PERIOD:
            return None
        self._force_tcp_last_t = now
        try:
            self._force_tcp.send(b'{"command":"get_force_data"}\r\n')
            buf = b""
            self._force_tcp.settimeout(1.0)
            while b"\r\n" not in buf and len(buf) < 16384:
                chunk = self._force_tcp.recv(4096)
                if not chunk:
                    break
                buf += chunk
            if not buf:
                return None
            line = buf.split(b"\r\n")[0].decode("utf-8", errors="replace")
            resp = json.loads(line)
            key = "zero_force_data" if SIX_FORCE_PREFER_ZERO else "force_data"
            fd = resp.get(key) or resp.get("force_data")
            if fd is None:
                fd = resp.get("zero_force_data")
            return None if fd is None else self._normalize_six(fd)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logging.warning("get_force_data TCP 查询失败: %s", e)
            return None

    def _filter_joint_deg_follow(self, joint_deg,
                                tau=0.10,
                                max_speed_deg_s=180.0,
                                deadband_deg=0.02,
                                q_deg=0.01):
        now = time.monotonic()
        dt = now - self._last_filter_t
        self._last_filter_t = now
        if dt <= 0:
            dt = 1e-3

        x = np.asarray(joint_deg, dtype=float)[:self.arm_axis]

        # init
        if self._filt_joint_deg is None:
            self._filt_joint_deg = x.copy()
        if self._last_sent_deg is None:
            self._last_sent_deg = self._filt_joint_deg.copy()

        # 1) 一阶低通
        alpha = dt / (tau + dt)
        y = (1 - alpha) * self._filt_joint_deg + alpha * x
        self._filt_joint_deg = y

        # 2) 限速
        max_delta = max_speed_deg_s * dt
        delta = np.clip(y - self._last_sent_deg, -max_delta, max_delta)
        candidate = self._last_sent_deg + delta

        # 3) 死区：小变化就“保持”，但不停止发送
        if np.max(np.abs(candidate - self._last_sent_deg)) < deadband_deg:
            candidate = self._last_sent_deg

        # 4) 精度限制：把指令量化到 q_deg（例如 0.01°）
        if q_deg is not None and q_deg > 0:
            candidate = np.round(candidate / q_deg) * q_deg

        self._last_sent_deg = candidate
        return candidate
    def _json_to_numpy(self, byte_data, key):
        """将字节数据解析为 NumPy 数组"""
        str_data = byte_data.decode("utf-8")
        try:
            data_list = json.loads(str_data)[key]
            if isinstance(data_list, dict):
                return data_list
        except KeyError:
            logging.error(f"Key '{key}' not found in JSON data")
            return np.array([])
        return np.array(data_list, dtype=float)

    def get_arm_data(self):
        """获取纯机械臂UDP实时数据"""
        try:
            data, addr = self.udp_socket.recvfrom(4096)
            if addr[0] != self.arm_ip:
                logging.warning(f"收到未知地址数据: {addr}")
                return None
            
            data = json.loads(data.decode('utf-8'))
            joint_angle = np.array(data['joint_status']['joint_position']) * 0.001
            joint_velocity = np.array(data['joint_status']['joint_speed']) * 0.001 if self.get_vel else None
            joint_torque = self.current_to_torque(np.array(data['joint_status']['joint_current']) / 1000000) if self.get_torque else None

            result = {'joint_angle': joint_angle}
            if joint_velocity is not None:
                result['joint_velocity'] = joint_velocity
            if joint_torque is not None:
                result['joint_torque'] = joint_torque
            if self.get_six_force:
                six = self._extract_six_force_from_udp(data)
                if six is not None:
                    result["six_force"] = six
                else:
                    self._udp_packets += 1
                    if (
                        not self._udp_force_warned
                        and self._udp_packets >= 30
                        and not SIX_FORCE_USE_TCP_QUERY
                    ):
                        self._udp_force_warned = True
                        logging.warning(
                            "UDP 中持续无六维力字段，请检查力传感器与 force_coordinate；"
                            "可设 SIX_FORCE_USE_TCP_QUERY=True。"
                            "最近一帧顶层键: %s",
                            list(data.keys()),
                        )

            return result
        except Exception as e:
            logging.error(f"读取机械臂数据出错: {e}")
            return None

    def _generate_command(self, data, cmd_type):
        """生成 JSON 命令"""
        data = np.array(data)
        if cmd_type == "joint":
            #data = np.floor(data * 1000).astype(int).tolist()
            data = np.round(data * 1000).astype(int).tolist()
            cmd = json.dumps({"command": "movej", "joint": data, "block": False, "v": 80, "r": 0}) + "\r\n"
        elif cmd_type == "gripper":
            data = data.astype(int).tolist()
            cmd = json.dumps({"command": "set_gripper_position", "position": data, "block": False}) + "\r\n"
        else:
            #data = np.floor(data * 1000).astype(int).tolist()
            data = np.round(data * 1000).astype(int).tolist()
            cmd = json.dumps({"command": "movej_canfd", "joint": data, "follow": True, "trajectory_mode":0}) + "\r\n"

        logging.debug(f"Generated command: {cmd}")
        return cmd

    def current_to_torque(self, current):
        """电流转扭矩"""
        return [c * k for c, k in zip(current, self.arm_ki)]
    def set_joint_canfd_position(self, joint_angle_deg):
        filt = self._filter_joint_deg_follow(
            joint_angle_deg,
            tau=0.10,
            max_speed_deg_s=180.0,
            deadband_deg=0.02,
            q_deg=0.001
        )
        cmd = self._generate_command(filt, "movej_canfd")
        self.arm.send(cmd.encode("utf-8"))
    def _filter_joint_deg(self, joint_deg, tau=0.10, max_speed_deg_s=180.0, deadband_deg=0.02):
        # movej 用：允许死区直接不发（返回 None）
        now = time.monotonic()
        dt = now - self._last_filter_t
        self._last_filter_t = now
        if dt <= 0:
            dt = 1e-3

        x = np.asarray(joint_deg, dtype=float)[:self.arm_axis]

        if self._filt_joint_deg is None:
            self._filt_joint_deg = x.copy()
        if self._last_sent_deg is None:
            self._last_sent_deg = self._filt_joint_deg.copy()

        alpha = dt / (tau + dt)
        self._filt_joint_deg = (1 - alpha) * self._filt_joint_deg + alpha * x

        max_delta = max_speed_deg_s * dt
        delta = np.clip(self._filt_joint_deg - self._last_sent_deg, -max_delta, max_delta)
        candidate = self._last_sent_deg + delta

        if np.max(np.abs(candidate - self._last_sent_deg)) < deadband_deg:
            return None

        self._last_sent_deg = candidate
        return candidate



    def set_joint_position(self, joint_angle):
        """设置机械臂的位置（接收角度值）"""
        try:
            if len(joint_angle) != self.arm_axis:
                logging.warning(f"角度维度不匹配！期望{self.arm_axis}轴，实际{len(joint_angle)}轴")
                joint_angle = joint_angle[:self.arm_axis]
            filt = self._filter_joint_deg(
                joint_angle,
                tau=0.08,              # 更稳：调大；更跟手：调小
                max_speed_deg_s=250.0, # 更稳：调小；更快：调大
                deadband_deg=0.03      # 更稳：调大；更细：调小
            )
            if filt is None:
                return

            cmd = self._generate_command(filt, cmd_type="joint")
            #cmd = self._generate_command(joint_angle, cmd_type="joint")
            self.arm.send(cmd.encode("utf-8"))
            _ = self.arm.recv(1024)
            _ = self.arm.recv(1024)
            logging.info(f"发送角度指令成功: {[round(a,3) for a in joint_angle]} °")
        except Exception as e:
            logging.error(f"发送角度指令失败: {e}")

    def close(self):
        """关闭连接"""
        if self._force_tcp is not None:
            try:
                self._force_tcp.close()
            except OSError:
                pass
            self._force_tcp = None
        self.arm.close()
        self.udp_socket.close()
        logging.info("✅ 连接已关闭")

# ===================== ROS2节点类 =====================
class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('rm_arm_ros2_subscriber')
                # ---------- QoS ----------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # 高频、尽量不丢包
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )# 订阅ROS2关节话题
        self.slave_gripper_pub = self.create_publisher(Float32, '/left_gripper_cmd', 10)
        self.slave_rm_pub = self.create_publisher(JointState, '/left/joint_states', 10)
        self.six_force_pub = None
        if GET_SIX_FORCE and Sixforce is not None:
            self.six_force_pub = self.create_publisher(Sixforce, SIX_FORCE_TOPIC, 10)
        elif GET_SIX_FORCE and Sixforce is None:
            self.get_logger().warning(
                "GET_SIX_FORCE=True 但无法导入 rm_ros_interfaces.msg.Sixforce，请 source 睿尔曼工作空间后再运行；已跳过六维力发布。"
            )
        self.subscription = self.create_subscription(
            JointState,
            ROS2_JOINT_TOPIC,
            self.joint_state_callback,
            10)  # QoS深度
        self.subscription  # 防止未使用变量警告
    def gripper_cal(self, msg:float):
        global ROS2_GRIPPER
        data_raw=msg
        if data_raw<0.0:
           data_raw=0.0        
        if data_raw>0.16:
           data_raw=0.16     
        inv=(data_raw-0.0)*1/(0.16-0.0)
        
        ROS2_GRIPPER = inv  
    def joint_state_callback(self, msg):
        global ROS2_JOINT_ANGLES_DEG
        joint_pos_rad = msg.position[:ARM_AXIS]
        ROS2_JOINT_ANGLES_DEG = np.array(joint_pos_rad, dtype=float) * 180 / np.pi
        #ROS2_JOINT_ANGLES_DEG = [round(deg, 3) for deg in joint_pos_deg]
        #ROS2_JOINT_ANGLES_DEG = joint_pos_deg  # 不要round
        if len(msg.position) > ARM_AXIS:
            self.gripper_cal(msg.position[ARM_AXIS])
        logging.debug(f"订阅到ROS2话题角度: {ROS2_JOINT_ANGLES_DEG} °")
    def joint_state_pub(self, position_deg, velocity=None, effort=None):
        rm_joint = JointState()
        rm_joint.header.stamp = self.get_clock().now().to_msg()
        rm_joint.name = [f"joint{i}" for i in range(1, 8)]
        rm_joint.position = position_deg
        if velocity is not None and len(velocity) >= ARM_AXIS:
            rm_joint.velocity = [float(v) for v in velocity[:ARM_AXIS]]
        else:
            rm_joint.velocity = [0.0] * ARM_AXIS
        if effort is not None and len(effort) >= ARM_AXIS:
            rm_joint.effort = [float(e) for e in effort[:ARM_AXIS]]
        else:
            rm_joint.effort = [0.0] * ARM_AXIS
        self.slave_rm_pub.publish(rm_joint)

    def publish_six_force(self, forces):
        if self.six_force_pub is None or forces is None:
            return
        f = np.asarray(forces, dtype=float).reshape(-1)
        if f.size < 6:
            return
        m = Sixforce()
        m.force_fx = float(f[0])
        m.force_fy = float(f[1])
        m.force_fz = float(f[2])
        m.force_mx = float(f[3])
        m.force_my = float(f[4])
        m.force_mz = float(f[5])
        self.six_force_pub.publish(m)

# ===================== 主逻辑 =====================
if __name__ == "__main__":
    # 1. 初始化ROS2
    rclpy.init()
    # 2. 创建ROS2订阅节点
    joint_sub = JointStateSubscriber()
    logging.info(f"✅ 已订阅ROS2话题: {ROS2_JOINT_TOPIC}")

    arm = None
    try:
        # 3. 初始化机械臂
        arm = RmArm()
        
        # 4. 循环：实时获取ROS2话题角度 → 控制机械臂
        logging.info("\n开始实时订阅ROS2话题并控制机械臂（按Ctrl+C退出）")
# ===== 控制参数：建议先这样 =====
        SEND_HZ = 100.0          # CANFD跟随建议 100Hz（10ms）
        LOG_HZ  = 10.0           # 日志 10Hz 就够了（避免刷屏影响时序）
        send_period = 1.0 / SEND_HZ
        log_period  = 1.0 / LOG_HZ

        next_send = time.monotonic()
        next_log  = time.monotonic()

        while rclpy.ok():
            # 1) ROS2回调：不要阻塞（timeout=0）
            rclpy.spin_once(joint_sub, timeout_sec=0.0)

            now = time.monotonic()

            # 2) 到点就发（固定周期）
            if now >= next_send:
                next_send += send_period

                if ROS2_JOINT_ANGLES_DEG is not None:
                    # ✅ 推荐：CANFD跟随（你要确保 set_joint_canfd_position 里“死区不停止发送”）
                    arm.set_joint_canfd_position(ROS2_JOINT_ANGLES_DEG)

                    # 如果你仍然想用 movej（不推荐高频），把上一行改成：
                    # arm.set_joint_position(ROS2_JOINT_ANGLES_DEG)
            if ROS2_GRIPPER is not None:
                msg_g = Float32()
                
                msg_g.data = float(ROS2_GRIPPER)
                joint_sub.slave_gripper_pub.publish(msg_g)
                
            # 3) 低频打印状态（不要每次都打印）
            if now >= next_log:
                next_log += send_period

                arm_data = arm.get_arm_data()
                sf = None
                if arm_data:
                    # 你这里 arm_data['joint_angle'] 实际是 “度”，别再当 rad 转换
                    joint_deg = [round(a, 3) for a in arm_data['joint_angle'][:ARM_AXIS]]
                    jv = arm_data.get('joint_velocity')
                    jt = arm_data.get('joint_torque')
                    joint_sub.joint_state_pub(joint_deg, velocity=jv, effort=jt)
                    sf = arm_data.get('six_force')
                if sf is None and GET_SIX_FORCE:
                    sf = arm.query_six_force_tcp()
                if sf is not None:
                    joint_sub.publish_six_force(sf)



            # 4) 防止空转占满CPU（很小的sleep，不影响周期）
            time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("\n收到退出信号，正在关闭...")
    except Exception as e:
        logging.error(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭机械臂连接
        if arm:
            arm.close()
        # 关闭ROS2节点
        joint_sub.destroy_node()
        rclpy.shutdown()
    logging.info("测试结束")

