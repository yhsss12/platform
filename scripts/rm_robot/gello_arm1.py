"""
Read 7-DOF master arm bus-servo counts -> convert to angles -> publish JointState
Publishes to /gello/main_arm/joint_states_right (matches xinzhubi_fk.py for 7-DOF)
"""

import math
import threading
import time
import os
import signal
import subprocess
from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import struct
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"


def kill_existing_instances(target_port=None):
    """Kill other running instances of this script to avoid serial conflicts.
    
    Args:
        target_port: Optional serial port path. If provided, processes using this port will also be killed.
    """
    my_pid = os.getpid()
    script_name = os.path.basename(__file__)
    pids_to_kill = set()

    # 1. Find processes by script name
    try:
        # Using pgrep -f to match full command line
        pids = subprocess.check_output(["pgrep", "-f", script_name]).decode().split()
        for pid_str in pids:
            pids_to_kill.add(int(pid_str))
    except subprocess.CalledProcessError:
        # No processes found by name
        pass

    # 2. Find processes using the serial port (if provided)
    if target_port and os.path.exists(target_port):
        try:
            # lsof -t returns just PIDs, one per line
            # -t: terse mode (only PIDs)
            pids = subprocess.check_output(["lsof", "-t", target_port], stderr=subprocess.DEVNULL).decode().split()
            for pid_str in pids:
                if pid_str.strip():
                    pids_to_kill.add(int(pid_str))
        except subprocess.CalledProcessError:
            # No processes found using the port
            pass
        except FileNotFoundError:
            # lsof might not be installed
            print("Warning: lsof not found, cannot check for processes using serial port.")

    # Kill them
    for pid in pids_to_kill:
        if pid != my_pid:
            print(f"Killing existing instance/process with PID: {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def count_to_rad(count: int) -> float:
    """Convert 12-bit count (0-4095) to radians with 2048 as zero."""
    deg = (count - 2048) * (180.0 / 2048.0)
    return math.radians(deg)


def normalize_gripper(val: int) -> float:
    """Normalize gripper value from [1967, 2564] to [0.0, 1.0]."""
    #min_val = 1967.0
    #max_val = 2564.0
    min_val = 2198.0
    max_val = 3385.0
    norm = (val - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, norm))

def read_frame(ser: serial.Serial) -> Optional[List[int]]:
    """Read one frame with header 0x55 0xAA and payload for 16 values (32 bytes).
    Format: 
    - Left Arm: 7 joints + 1 gripper
    - Right Arm: 7 joints + 1 gripper
    Total 16 uint16 values.
    """
    # sync to header
    while True:
        b1 = ser.read(1)
        if not b1:
            return None
        if b1 == b'\x55':
            b2 = ser.read(1)
            if b2 == b'\xAA':
                break
    # Read 32 bytes for 16 values (each value is 2 bytes)
    payload = ser.read(32)
    if len(payload) != 32:
        return None
    vals = []
    for i in range(0, 32, 2):
        # Interpret as Big-Endian Signed 16-bit integer
        v = int.from_bytes(payload[i:i+2], byteorder='big', signed=True)
        vals.append(v)
    return vals


class SerialToJointStateNodeDualArm(Node):
    def __init__(self):
        super().__init__("serial_to_joint_state_dual_arm")
        # Update port to use by-id as requested
        # self.declare_parameter("port", "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0DVETY-if00-port0")
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("baud", 115200)
        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baud = int(self.get_parameter("baud").get_parameter_value().integer_value)

        # Left arm joint names
        self.left_joint_names = [
            "left_joint1", "left_joint2", "left_joint3", "left_joint4",
            "left_joint5", "left_joint6", "left_joint7", "left_gripper"
        ]
        
        # Right arm joint names
        self.right_joint_names = [
            "right_joint1", "right_joint2", "right_joint3", "right_joint4",
            "right_joint5", "right_joint6", "right_joint7", "right_gripper"
        ]

        # Publishers for both arms
        self.pub_left = self.create_publisher(
            JointState, "/gello/left_arm/joint_states", 10
        )
        self.pub_right = self.create_publisher(
            JointState, "/gello/right_arm/joint_states", 10
        )

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.get_logger().info(f"Serial opened: {self.port} @ {self.baud}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial {self.port}: {e}")
            raise

        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            vals = read_frame(self.ser)
            if vals is None:
                continue
            
            # Expected 16 values: 
            # 0-6: Left Arm Joints
            # 7: Left Gripper
            # 8-14: Right Arm Joints
            # 15: Right Gripper
            
            if len(vals) != 16:
                continue

            # Process Left Arm
            left_joints = vals[0:7]
            left_gripper = vals[7]
            left_rad = [count_to_rad(v) for v in left_joints]
            # Inverse joint2 if needed (keeping original logic for joint2)
            left_rad[0] = -(left_rad[0]+math.pi)
            # left_rad[1] = -left_rad[1]
            left_rad[2] = -left_rad[2]
            left_rad[4] = -left_rad[4]
            left_rad[6] = -left_rad[6]
            # Append normalized gripper value
            left_rad.append(normalize_gripper(left_gripper))
            
            msg_left = JointState()
            msg_left.header.stamp = self.get_clock().now().to_msg()
            msg_left.name = self.left_joint_names
            msg_left.position = left_rad
            self.pub_left.publish(msg_left)

            # Process Right Arm
            right_joints = vals[8:15]
            right_gripper = vals[15]
            right_rad = [count_to_rad(v) for v in right_joints]
            # Inverse joint2 if needed
            #right_rad[0] = -(right_rad[0]+math.pi) 
            right_rad[0] = -right_rad[0]+math.pi
            # right_rad[1] = -right_rad[1]
            right_rad[2] = -right_rad[2]
            #right_rad[4] = -right_rad[4]+0.3160000423044421
            right_rad[4] = -right_rad[4]
            right_rad[6] = -right_rad[6]
            # Append normalized gripper value
            right_rad.append(normalize_gripper(right_gripper))

            msg_right = JointState()
            msg_right.header.stamp = self.get_clock().now().to_msg()
            msg_right.name = self.right_joint_names
            msg_right.position = right_rad
            self.pub_right.publish(msg_right)

            # Optional console debug (print first few joints of each arm)
            # print(f"L_raw: {left_raw} \nL_deg: {[round(math.degrees(d), 2) for d in left_rad]}")
            # print(f"R_raw: {right_raw} \nR_deg: {[round(math.degrees(d), 2) for d in right_rad]}")
            # print("-" * 30)

    def destroy_node(self):
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    kill_existing_instances(DEFAULT_PORT)
    rclpy.init()
    node = SerialToJointStateNodeDualArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

