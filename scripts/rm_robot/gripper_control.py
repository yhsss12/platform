#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# Pymodbus compatibility wrapper
try:
    from pymodbus.client import ModbusSerialClient
    PYMODBUS_V3 = True
except ImportError:
    from pymodbus.client.sync import ModbusSerialClient
    PYMODBUS_V3 = False

class GripperDevice:
    def __init__(self, name, port, node, reverse=True):
        self.name = name
        self.port = port
        self.node = node
        self.reverse = reverse
        self.unit = 1
        self.HW_MAX_POS = 9000
        #self.HW_MAX_POS = 1000
        self.ROS_MAX_POS = 1.0
        self.connected = False
        self.client = None

        self.node.get_logger().info(f"[{self.name}] Connecting to {self.port} ...")
        
        # Init client
        if PYMODBUS_V3:
            self.client = ModbusSerialClient(port=self.port, baudrate=115200, 
                                           parity='N', stopbits=1, bytesize=8, timeout=0.05)
        else:
            self.client = ModbusSerialClient(method='rtu', port=self.port, baudrate=115200, 
                                           parity='N', stopbits=1, bytesize=8, timeout=0.05)
            
        if self.client.connect():
            self.connected = True
            self.node.get_logger().info(f"[{self.name}] Serial connected.")
            try:
                self.enable()
                time.sleep(0.2)
                self.setup_params()
            except Exception as e:
                self.node.get_logger().error(f"[{self.name}] Init failed: {e}")
        else:
            self.node.get_logger().error(f"[{self.name}] Failed to connect serial: {self.port}")

    def _write_register(self, address, value):
        if PYMODBUS_V3:
            return self.client.write_register(address, value, device_id=self.unit)
        else:
            return self.client.write_register(address, value, unit=self.unit)

    def _read_holding_registers(self, address, count):
        if PYMODBUS_V3:
            return self.client.read_holding_registers(address, count=count, device_id=self.unit)
        else:
            return self.client.read_holding_registers(address, count=count, unit=self.unit)

    def write_reg(self, addr, value):
        if not self.connected: return
        try:
            result = self._write_register(addr, value)
            if hasattr(result, 'isError') and result.isError():
                self.node.get_logger().warn(f"[{self.name}] Write failed {hex(addr)} -> {value}")
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] Write exception {hex(addr)}: {e}")

    def enable(self):
        self.write_reg(0x0100, 1)
        self.node.get_logger().info(f"[{self.name}] Enabled")

    def setup_params(self):
        self.write_reg(0x0104, 100)   # Speed
        self.write_reg(0x0105, 100)   # Force
        self.write_reg(0x0106, 1000)  # Accel
        self.write_reg(0x0107, 1000)  # Decel
        self.node.get_logger().info(f"[{self.name}] Params set")

    def set_position(self, ros_pos):
        if not self.connected: return
        
        # Logic
        if self.reverse:
            target = self.ROS_MAX_POS - ros_pos
        else:
            target = ros_pos
        
        # Clamp target 0-1 before mapping to HW
        target = max(0.0, min(1.0, target))
        
        hw_pos = int((target / self.ROS_MAX_POS) * self.HW_MAX_POS)
        
        # Write high/low
        high = (hw_pos >> 16) & 0xFFFF
        low = hw_pos & 0xFFFF
        try:
            self._write_register(0x0102, high)
            self._write_register(0x0103, low)
            self.write_reg(0x0108, 1) # Trigger move
            self.node.get_logger().info(f"[{self.name}] Cmd {ros_pos:.2f} -> HW {hw_pos}")
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] Move exception: {e}")
            
    def get_feedback(self):
        if not self.connected: return None
        try:
            result = self._read_holding_registers(0x0418, 2)
            if hasattr(result, 'isError') and result.isError():
                # self.node.get_logger().warn(f"[{self.name}] Read feedback error")
                return None
            if not hasattr(result, 'registers'):
                return None
            
            high, low = result.registers
            pos_fb = (high << 16) | low
            
            # Map
            pos_fb_clamped = max(0, min(pos_fb, self.HW_MAX_POS))
            pos_norm = (pos_fb_clamped / self.HW_MAX_POS) * self.ROS_MAX_POS
            
            if self.reverse:
                pos_final = self.ROS_MAX_POS - pos_norm
            else:
                pos_final = pos_norm
                
            return pos_final
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] Feedback exception: {e}")
            return None
            
    def close(self):
        if self.client:
            self.client.close()


class DualGripperNode(Node):
    def __init__(self):
        super().__init__('gripper_control')
        
        # Parameters
        # Right Gripper
        self.declare_parameter('left_port', '/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0DVETY-if00-port0')
        # Left Gripper
        self.declare_parameter('right_port', '/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0DVNBK-if00-port0')
        
        self.declare_parameter('reverse', True)
        self.declare_parameter('feedback_period', 0.02)
        
        right_port = self.get_parameter('right_port').get_parameter_value().string_value
        left_port = self.get_parameter('left_port').get_parameter_value().string_value
        reverse = self.get_parameter('reverse').get_parameter_value().bool_value
        self.feedback_period = self.get_parameter('feedback_period').get_parameter_value().double_value
        
        self.get_logger().info(f"Starting Dual Gripper Node...")
        self.get_logger().info(f"Right Port: {right_port}")
        self.get_logger().info(f"Left Port:  {left_port}")
        self.get_logger().info(f"Reverse:    {reverse}")

        # Initialize Devices
        self.right_gripper = GripperDevice("Right", right_port, self, reverse)
        self.left_gripper = GripperDevice("Left", left_port, self, reverse)
        
        # Publishers & Subscribers
        # Right
        self.pub_right = self.create_publisher(Float32, '/right_gripper_state', 10)
        self.sub_right = self.create_subscription(Float32, '/right_gripper_cmd', self.right_cmd_cb, 10)
        
        # Left
        self.pub_left = self.create_publisher(Float32, '/left_gripper_state', 10)
        self.sub_left = self.create_subscription(Float32, '/left_gripper_cmd', self.left_cmd_cb, 10)
        
        # Timer for feedback
        self.timer = self.create_timer(self.feedback_period, self.timer_cb)
        self.get_logger().info("Ready. Topics: /right_gripper_*, /left_gripper_*")

    def right_cmd_cb(self, msg: Float32):
        pos = float(msg.data)
        if 0.0 <= pos <= 1.0:
            self.right_gripper.set_position(pos)
            self.publish_feedback(self.right_gripper, self.pub_right)
        else:
            self.get_logger().warn(f"[Right] Invalid cmd: {pos}")

    def left_cmd_cb(self, msg: Float32):
        pos = float(msg.data)
        if 0.0 <= pos <= 1.0:
            self.left_gripper.set_position(pos)
            self.publish_feedback(self.left_gripper, self.pub_left)
        else:
            self.get_logger().warn(f"[Left] Invalid cmd: {pos}")

    def timer_cb(self):
        self.publish_feedback(self.right_gripper, self.pub_right)
        self.publish_feedback(self.left_gripper, self.pub_left)
        
    def publish_feedback(self, device, pub):
        val = device.get_feedback()
        if val is not None:
            msg = Float32()
            msg.data = float(val)
            pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DualGripperNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node exception: {e}")
    finally:
        if node:
            if hasattr(node, 'right_gripper'): node.right_gripper.close()
            if hasattr(node, 'left_gripper'): node.left_gripper.close()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
