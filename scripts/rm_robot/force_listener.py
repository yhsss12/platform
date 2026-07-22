# force_listener.py
#!/usr/bin/env python3
"""
六维力数据监听节点
订阅睿尔曼机械臂的六维力话题，可转发或处理数据
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from rm_ros_interfaces.msg import Sixforce


class ForceListener(Node):
    def __init__(self):
        super().__init__('force_listener')
        
        # 订阅左臂六维力
        self.left_force_sub = self.create_subscription(
            Sixforce,
            '/left/rm_driver/get_force_data_result',
            self.left_force_callback,
            10
        )
        
        # 订阅右臂六维力
        self.right_force_sub = self.create_subscription(
            Sixforce,
            '/right/rm_driver/get_force_data_result',
            self.right_force_callback,
            10
        )
        
        # 可以发布处理后的力数据
        self.left_force_pub = self.create_publisher(
            Float64MultiArray,
            '/processed/left_force',
            10
        )
        self.right_force_pub = self.create_publisher(
            Float64MultiArray,
            '/processed/right_force',
            10
        )
        
        self.left_force = [0.0] * 6
        self.right_force = [0.0] * 6
        
        self.get_logger().info('六维力监听节点已启动')
    
    def left_force_callback(self, msg):
        self.left_force = [
            msg.force_fx, msg.force_fy, msg.force_fz,
            msg.force_mx, msg.force_my, msg.force_mz,
        ]
        # 处理或转发...
        self.publish_processed(self.left_force_pub, self.left_force)
    
    def right_force_callback(self, msg):
        self.right_force = [
            msg.force_fx, msg.force_fy, msg.force_fz,
            msg.force_mx, msg.force_my, msg.force_mz,
        ]
        self.publish_processed(self.right_force_pub, self.right_force)
    
    def publish_processed(self, pub, force_data):
        msg = Float64MultiArray()
        msg.data = force_data
        pub.publish(msg)


def main():
    rclpy.init()
    node = ForceListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()