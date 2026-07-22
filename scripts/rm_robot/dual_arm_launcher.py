#!/usr/bin/env python3
"""
主启动程序：按顺序启动双机械臂控制系统的四个核心程序
启动顺序：
1. gello_arm1.py (读取主臂数据并发布joint_states)
2. gripper_control.py (控制夹爪)
3. realman_left_arm.py (控制左从臂)
4. realman_right_arm.py (控制右从臂)
"""

import subprocess
import time
import signal
import sys
import os
import atexit

class DualArmLauncher:
    def __init__(self):
        self.processes = []
        self.process_names = [
            "gello_arm1.py",
            "gripper_control.py", 
            "realman_left_arm.py",
            "realman_right_arm.py"
        ]
        
        # 注册退出处理函数
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def dump_recent_output(self, process_info, max_lines=80):
        process = process_info["process"]
        name = process_info["name"]

        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                process_info["stdout_lines"].append(line)
        except Exception:
            pass

        lines = process_info.get("stdout_lines", [])
        if not lines:
            return

        tail = lines[-max_lines:]
        print(f"\n--- {name} 输出(最近 {len(tail)} 行) ---")
        for line in tail:
            print(f"[{name}] {line.rstrip()}")
        print(f"--- {name} 输出结束 ---\n")

    def start_process(self, script_name, delay=2.0):
        """启动单个Python脚本"""
        print(f"\n=== 正在启动: {script_name} ===")
        
        try:
            # 使用python3执行脚本
            process = subprocess.Popen(
                ['python3', script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append({
                'name': script_name,
                'process': process,
                'stdout_lines': []
            })
            
            print(f"✅ {script_name} 已启动 (PID: {process.pid})")
            
            # 等待一段时间确保程序初始化完成
            print(f"等待 {delay} 秒让 {script_name} 初始化...")
            time.sleep(delay)
            
            # 检查进程是否仍在运行
            if process.poll() is not None:
                print(f"❌ {script_name} 启动失败，退出码: {process.returncode}")
                # 读取输出以便调试
                stdout, _ = process.communicate(timeout=1)
                if stdout:
                    print(f"输出信息:\n{stdout}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 启动 {script_name} 时发生错误: {e}")
            return False
    
    def monitor_output(self, process_info):
        """非阻塞方式读取进程输出"""
        process = process_info['process']
        name = process_info['name']
        
        try:
            # 尝试读取一行输出（非阻塞）
            import fcntl
            import os
            
            fd = process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            
            try:
                line = process.stdout.readline()
                if line:
                    print(f"[{name}] {line.rstrip()}")
                    process_info['stdout_lines'].append(line)
            except (IOError, BlockingIOError):
                pass  # 没有输出可用
                
        except Exception as e:
            # 如果无法设置非阻塞，忽略
            pass
    
    def start_all(self):
        """按顺序启动所有程序"""
        print("=" * 60)
        print("🚀 双机械臂控制系统启动程序")
        print("=" * 60)
        
        # 检查当前目录下所有必要的脚本是否存在
        missing_scripts = []
        for script in self.process_names:
            if not os.path.exists(script):
                missing_scripts.append(script)
        
        if missing_scripts:
            print("❌ 以下脚本文件不存在:")
            for script in missing_scripts:
                print(f"   - {script}")
            print("\n请确保所有脚本都在当前目录下")
            return False
        
        # 1. 启动主臂读取程序
        if not self.start_process("gello_arm1.py", delay=3):
            return False
        
        # 2. 启动夹爪控制程序
        if not self.start_process("gripper_control.py", delay=3):
            return False
        
        # 3. 启动左从臂控制程序
        if not self.start_process("realman_left_arm.py", delay=3):
            return False
        
        # 4. 启动右从臂控制程序
        if not self.start_process("realman_right_arm.py", delay=3):
            return False
        
        print("\n" + "=" * 60)
        print("✅ 所有程序已启动！")
        print("=" * 60)
        print("\n运行状态监控:")
        print("-" * 40)
        
        return True
    
    def run(self):
        """主运行循环"""
        if not self.start_all():
            return
        
        try:
            # 主监控循环
            while True:
                # 检查所有进程状态
                all_running = True
                for proc_info in self.processes:
                    process = proc_info['process']
                    name = proc_info['name']
                    
                    # 检查进程是否仍在运行
                    if process.poll() is not None:
                        print(f"❌ {name} 已停止，退出码: {process.returncode}")
                        self.dump_recent_output(proc_info)
                        all_running = False
                    
                    # 监控输出
                    self.monitor_output(proc_info)
                
                if not all_running:
                    print("\n⚠️ 检测到有程序停止运行，正在关闭所有进程...")
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，正在关闭所有程序...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理所有子进程"""
        if not self.processes:
            return
        
        print("\n正在关闭所有程序...")
        
        for proc_info in self.processes:
            name = proc_info['name']
            process = proc_info['process']
            
            if process.poll() is None:  # 进程仍在运行
                print(f"正在终止 {name} (PID: {process.pid})...")
                try:
                    # 先尝试优雅终止
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # 强制结束
                        print(f"强制结束 {name}...")
                        process.kill()
                        process.wait()
                except Exception as e:
                    print(f"关闭 {name} 时出错: {e}")
        
        self.processes.clear()
        print("所有程序已关闭")
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        print(f"\n收到信号 {signum}")
        self.cleanup()
        sys.exit(0)


def main():
    launcher = DualArmLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
