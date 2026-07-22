#!/usr/bin/env python3

import sys
import logging
import numpy as np
import h5py
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import cv2
import tqdm
import os
import glob
from dataclasses import dataclass
import time
import tracemalloc
import gc
import re
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量控制详细日志输出
verbose_mode = False

def log_info(message: str, verbose_only: bool = False):
    """控制日志输出，verbose_only=True时只在详细模式下输出"""
    if not verbose_only or verbose_mode:
        logger.info(message)

def log_debug(message: str):
    """调试日志，只在详细模式下输出"""
    if verbose_mode:
        logger.debug(message)

def log_warning(message: str, verbose_only: bool = False):
    """警告日志，verbose_only=True时只在详细模式下输出"""
    if not verbose_only or verbose_mode:
        logger.warning(message)

class PerformanceMonitor:
    """性能监控类，用于跟踪处理时间、内存使用等指标"""
    
    def __init__(self):
        self.timings = {}
        self.memory_snapshots = []
        self.cpu_snapshots = []
        self.total_messages = 0  # 记录总消息数用于计算处理速度
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        self.start_time = None
        self.tracemalloc_started = False
        
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
        self._record_memory()
    
    def mark(self, label: str):
        """标记时间点"""
        current_time = time.time()
        if self.start_time is not None:
            elapsed = current_time - (self.timings.get(label + '_start', self.start_time))
            self.timings[label] = elapsed
            self.timings[label + '_start'] = current_time
        self._record_memory(label)
    
    def _record_memory(self, label: str = None):
        """记录内存使用"""
        try:
            # 获取进程内存使用
            if self.process is not None:
                mem_info = self.process.memory_info()
                rss_mb = mem_info.rss / 1024 / 1024  # RSS in MB
            else:
                rss_mb = 0
            
            # 获取tracemalloc统计
            if self.tracemalloc_started:
                current, peak = tracemalloc.get_traced_memory()
                current_mb = current / 1024 / 1024
                peak_mb = peak / 1024 / 1024
            else:
                current_mb = 0
                peak_mb = 0
            
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'rss_mb': rss_mb,
                'tracemalloc_current_mb': current_mb,
                'tracemalloc_peak_mb': peak_mb
            }
            self.memory_snapshots.append(snapshot)
            
            # 记录CPU使用率
            if self.process is not None:
                try:
                    cpu_percent = self.process.cpu_percent(interval=None)
                    self.cpu_snapshots.append({
                        'label': label,
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent
                    })
                except Exception:
                    pass
        except Exception as e:
            log_debug(f"记录内存使用失败: {e}")
    
    def get_file_size(self, filepath: str) -> float:
        """获取文件大小（MB）"""
        try:
            if os.path.exists(filepath):
                return os.path.getsize(filepath) / 1024 / 1024
            return 0.0
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计信息"""
        if not self.memory_snapshots:
            return {'peak_rss_mb': 0, 'avg_rss_mb': 0, 'peak_tracemalloc_mb': 0}
        
        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        tracemalloc_peaks = [s['tracemalloc_peak_mb'] for s in self.memory_snapshots]
        
        return {
            'peak_rss_mb': max(rss_values),
            'avg_rss_mb': np.mean(rss_values),
            'min_rss_mb': min(rss_values),
            'peak_tracemalloc_mb': max(tracemalloc_peaks) if tracemalloc_peaks else 0
        }
    
    def calculate_alignment_metrics(self, alignment_details: Dict, dropped_counts: Dict, 
                                   total_timestamps: int) -> Dict[str, Any]:
        """计算对齐精度指标"""
        metrics = {
            'topics': {},
            'overall': {}
        }
        
        all_time_diffs = []
        all_abs_time_diffs = []
        total_dropped = 0
        total_aligned = 0
        
        for topic_name, details in alignment_details.items():
            if not details:
                continue
            
            # 过滤掉没有时间差的数据（可能是警告）
            valid_details = [d for d in details if d.get('time_diff') is not None]
            if not valid_details:
                continue
            
            time_diffs = [d['time_diff'] for d in valid_details]
            abs_time_diffs = [d['abs_time_diff'] for d in valid_details]
            
            topic_metrics = {
                'aligned_count': len(valid_details),
                'dropped_count': dropped_counts.get(topic_name, 0),
                'mean_time_diff_ms': np.mean(time_diffs) * 1000,
                'mean_abs_time_diff_ms': np.mean(abs_time_diffs) * 1000,
                'max_time_diff_ms': max(time_diffs) * 1000,
                'min_time_diff_ms': min(time_diffs) * 1000,
                'std_time_diff_ms': np.std(time_diffs) * 1000,
                'max_abs_time_diff_ms': max(abs_time_diffs) * 1000,
                'min_abs_time_diff_ms': min(abs_time_diffs) * 1000,
            }
            
            if total_timestamps > 0:
                topic_metrics['success_rate'] = len(valid_details) / total_timestamps * 100
            else:
                topic_metrics['success_rate'] = 0.0
            
            metrics['topics'][topic_name] = topic_metrics
            
            all_time_diffs.extend(time_diffs)
            all_abs_time_diffs.extend(abs_time_diffs)
            total_dropped += dropped_counts.get(topic_name, 0)
            total_aligned += len(valid_details)
        
        # 计算整体指标
        if all_time_diffs:
            metrics['overall'] = {
                'total_aligned': total_aligned,
                'total_dropped': total_dropped,
                'mean_time_diff_ms': np.mean(all_time_diffs) * 1000,
                'mean_abs_time_diff_ms': np.mean(all_abs_time_diffs) * 1000,
                'max_time_diff_ms': max(all_time_diffs) * 1000,
                'min_time_diff_ms': min(all_time_diffs) * 1000,
                'std_time_diff_ms': np.std(all_time_diffs) * 1000,
                'max_abs_time_diff_ms': max(all_abs_time_diffs) * 1000,
                'min_abs_time_diff_ms': min(all_abs_time_diffs) * 1000,
            }
            
            if total_timestamps > 0:
                metrics['overall']['overall_success_rate'] = total_aligned / (total_timestamps * len(alignment_details)) * 100
            else:
                metrics['overall']['overall_success_rate'] = 0.0
        else:
            metrics['overall'] = {
                'total_aligned': 0,
                'total_dropped': total_dropped,
                'mean_time_diff_ms': 0,
                'mean_abs_time_diff_ms': 0,
            }
        
        return metrics
    
    def print_metrics(self, input_file: str, output_file: str, alignment_metrics: Dict = None):
        """打印简化的性能指标：处理速度、内存消耗、CPU消耗"""
        total_time = time.time() - self.start_time if self.start_time else 0
        memory_stats = self.get_memory_stats()
        
        print("\n" + "=" * 80)
        print("转换性能指标报告")
        print("=" * 80)
        
        # 处理速度指标（msgs/s）
        print(f"\n【处理速度】")
        if self.total_messages > 0 and total_time > 0:
            msg_rate = self.total_messages / total_time
            print(f"  处理速度: {msg_rate:.2f} msgs/s")
            print(f"  总消息数: {self.total_messages}")
            print(f"  总处理时间: {total_time:.2f} 秒")
        else:
            print(f"  总处理时间: {total_time:.2f} 秒")
        
        # 内存消耗指标
        print(f"\n【内存消耗】")
        if not PSUTIL_AVAILABLE:
            print("  注意: psutil未安装，内存统计不可用")
        else:
            if memory_stats['peak_rss_mb'] > 0:
                print(f"  峰值内存: {memory_stats['peak_rss_mb']:.2f} MB")
                print(f"  平均内存: {memory_stats['avg_rss_mb']:.2f} MB")
            else:
                print("  内存统计: 无数据")
        
        # CPU消耗指标
        print(f"\n【CPU消耗】")
        if not PSUTIL_AVAILABLE:
            print("  注意: psutil未安装，CPU统计不可用")
        else:
            if self.cpu_snapshots:
                cpu_values = [s['cpu_percent'] for s in self.cpu_snapshots if s.get('cpu_percent') is not None]
                if cpu_values:
                    avg_cpu = np.mean(cpu_values)
                    peak_cpu = max(cpu_values)
                    print(f"  平均CPU使用率: {avg_cpu:.2f}%")
                    print(f"  峰值CPU使用率: {peak_cpu:.2f}%")
                else:
                    print("  CPU统计: 无数据")
            else:
                print("  CPU统计: 无数据")
        
        print("=" * 80 + "\n")

@dataclass
class TopicConfig:
    """话题配置类"""
    topic_name: str
    message_type: str
    hdf5_path: str
    data_type: str = 'float32'
    description: str = ""
    custom_processor: Optional[str] = None
    custom_params: Dict[str, Any] = None
    # 新增：支持话题名称模式匹配
    topic_patterns: Optional[List[str]] = None  # 支持多个可能的话题名称模式
    auto_detect_type: bool = False  # 是否自动检测消息类型
    force_create_subdatasets: bool = False  # 是否强制创建缺失的子数据集（使用NaN填充）

@dataclass
class AlignmentConfig:
    """时间对齐配置类"""
    main_timeline_topic: Optional[str] = None  # None表示自动选择数据量最小的
    alignment_window: float = 0.05  # 对齐窗口大小（秒）
    target_fps: float = 30.0  # 目标帧率
    sample_drop: int = 0  # 丢弃首尾帧数
    relative_start: bool = False  # 相对起始点
    delta_action: bool = False  # 增量动作
    # 只保留 backfill_on_grid 策略
    strategy: str = "backfill_on_grid"
    grid_fps: float = 15.0
    expand_start: bool = False
    # 目标数据时长（秒），None表示不限制
    target_duration: Optional[float] = None


def _alignment_window_size(alignment_config: AlignmentConfig) -> float:
    """与 backfill_on_grid 成功路径一致的窗口步长（秒）。"""
    fps = (
        alignment_config.grid_fps
        if alignment_config.grid_fps and alignment_config.grid_fps > 0
        else 15.0
    )
    return 1.0 / fps


class FlexibleMcapProcessor:
    """灵活的消息处理器"""
    
    def __init__(self):
        self.processors = {
            'joint_state': self._process_joint_state,
            'compressed_image': self._process_compressed_image,
            'image': self._process_image,
            'float64_multiarray': self._process_float64_multiarray,
            'float32_multiarray': self._process_float32_multiarray,
            'float32': self._process_float32,
            'float64': self._process_float64,
            'sixforce': self._process_sixforce,
            'twist': self._process_twist,
            'twist_stamped': self._process_twist_stamped,
            'pose_stamped': self._process_pose_stamped,
            'int32': self._process_int32,
            'depth_image': self._process_depth_image,
            'custom': self._process_custom
        }
    
    def _process_joint_state(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理关节状态消息 - 分别处理各字段，保留各自维度"""
        try:
            # 获取关节数量配置
            # joint_count > 0: 作为目标维度（不足时填充0，超出时截取）
            # joint_count <= 0 或未配置: 使用实际数据的维度（自动适应）
            joint_count = config.custom_params.get('joint_count', -1) if config.custom_params else -1
            
            # 处理位置数据 - 保持原始维度
            position_data = None
            position_dof = 0
            if hasattr(msg, 'position') and msg.position is not None:
                position_data = np.array(msg.position, dtype=getattr(np, config.data_type))
                position_dof = len(position_data)
                # 如果配置了joint_count且>0，进行填充或截取
                if joint_count > 0:
                    if position_dof < joint_count:
                        padding = np.zeros(joint_count - position_dof, dtype=getattr(np, config.data_type))
                        position_data = np.concatenate([position_data, padding])
                        position_dof = joint_count
                    elif position_dof > joint_count:
                        position_data = position_data[:joint_count]
                        position_dof = joint_count
            
            # 处理速度数据 - 保持原始维度
            velocity_data = None
            velocity_dof = 0
            if hasattr(msg, 'velocity') and msg.velocity and len(msg.velocity) > 0:
                velocity_data = np.array(msg.velocity, dtype=getattr(np, config.data_type))
                velocity_dof = len(velocity_data)
                # 如果配置了joint_count且>0，进行填充或截取
                if joint_count > 0:
                    if velocity_dof < joint_count:
                        padding = np.zeros(joint_count - velocity_dof, dtype=getattr(np, config.data_type))
                        velocity_data = np.concatenate([velocity_data, padding])
                        velocity_dof = joint_count
                    elif velocity_dof > joint_count:
                        velocity_data = velocity_data[:joint_count]
                        velocity_dof = joint_count
            
            # 处理力矩数据 - 保持原始维度
            effort_data = None
            effort_dof = 0
            if hasattr(msg, 'effort') and msg.effort and len(msg.effort) > 0:
                effort_data = np.array(msg.effort, dtype=getattr(np, config.data_type))
                effort_dof = len(effort_data)
                # 如果配置了joint_count且>0，进行填充或截取
                if joint_count > 0:
                    if effort_dof < joint_count:
                        padding = np.zeros(joint_count - effort_dof, dtype=getattr(np, config.data_type))
                        effort_data = np.concatenate([effort_data, padding])
                        effort_dof = joint_count
                    elif effort_dof > joint_count:
                        effort_data = effort_data[:joint_count]
                        effort_dof = joint_count
            
            # 如果所有字段都为空，返回None
            if position_data is None and velocity_data is None and effort_data is None:
                log_warning(f"关节状态消息所有字段都为空: {config.topic_name}", verbose_only=True)
                return None
            
            return {
                'data': position_data,
                'velocity': velocity_data,
                'effort': effort_data,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'joint_names': list(msg.name) if hasattr(msg, 'name') else [],
                'position_dof': position_dof,  # 记录位置维度
                'velocity_dof': velocity_dof,  # 记录速度维度
                'effort_dof': effort_dof       # 记录力矩维度
            }
        except Exception as e:
            log_warning(f"关节状态处理失败 {config.topic_name}: {e}")
            import traceback
            log_debug(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def _process_compressed_image(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理压缩图像消息"""
        try:
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                # 若按压缩图像解码失败，尝试按原始彩色图像处理（topic类型变更的兼容）
                if hasattr(msg, 'height') and hasattr(msg, 'width') and hasattr(msg, 'encoding'):
                    return self._process_image(msg, config)
                raise ValueError("Failed to decode compressed image")
            
            # 默认执行 BGR->RGB，仅在显式设置 bgr_to_rgb=false 时关闭
            if (config.custom_params is None) or config.custom_params.get('bgr_to_rgb', True):
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # 根据配置调整图像尺寸
            if config.custom_params and 'resize' in config.custom_params:
                target_size = config.custom_params['resize']
                if len(target_size) == 2:
                    cv_img = cv2.resize(cv_img, (target_size[1], target_size[0]))  # (width, height)
            
            return {
                'data': cv_img,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'format': msg.format if hasattr(msg, 'format') else 'unknown',
                'original_shape': cv_img.shape
            }
        except Exception as e:
            log_debug(f"压缩图像处理失败: {e}")
            return None
    
    def _process_image(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理原始图像消息"""
        try:
            height, width = msg.height, msg.width
            encoding = getattr(msg, 'encoding', 'rgb8')
            data_buf = np.frombuffer(msg.data, dtype=np.uint8)

            if encoding == 'rgb8':
                rgb_array = data_buf.reshape((height, width, 3))
            elif encoding == 'bgr8':
                bgr_array = data_buf.reshape((height, width, 3))
                rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
            elif encoding == 'rgba8':
                rgba = data_buf.reshape((height, width, 4))
                rgb_array = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
            elif encoding == 'bgra8':
                bgra = data_buf.reshape((height, width, 4))
                rgb_array = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            elif encoding == 'mono8':
                gray = data_buf.reshape((height, width))
                rgb_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif encoding in ('mono16', '16UC1', '16UC3', '16UC4'):
                # 深度图像（uint16），保持单通道结构
                depth_dtype = np.uint16
                depth_array = np.frombuffer(msg.data, dtype=depth_dtype).reshape((height, width))
                # 对于深度数据，我们不强制转换为RGB，直接返回数据
                # 将键名设为 'depth'，以便在 convert_to_hdf5 中被正确识别并保存到 depth 路径
                return {
                    'depth': depth_array.astype(depth_dtype),
                    'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                    'encoding': encoding
                }
            else:
                log_warning(f"不支持的图像编码: {encoding}", verbose_only=True)
                return None
            
            # 根据配置调整图像尺寸
            if config.custom_params and 'resize' in config.custom_params:
                target_size = config.custom_params['resize']
                if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                    rgb_array = cv2.resize(rgb_array, (target_size[1], target_size[0]))
            
            return {
                'data': rgb_array,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'encoding': encoding
            }
        except Exception as e:
            log_debug(f"原始图像处理失败: {e}")
            return None
    
    def _process_float64_multiarray(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理Float64MultiArray消息"""
        try:
            return {
                'data': np.array(msg.data, dtype=np.float64),
                'timestamp': 0.0,  # MultiArray通常没有时间戳
                'layout': {
                    'dim': [{'label': dim.label, 'size': dim.size, 'stride': dim.stride} 
                           for dim in msg.layout.dim] if hasattr(msg, 'layout') else []
                }
            }
        except Exception as e:
            log_debug(f"Float64MultiArray处理失败: {e}")
            return None

    def _process_sixforce(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理 rm_ros_interfaces/msg/Sixforce 消息

        期望字段（realman_*_arm.py / force_listener.py 已使用）：
          - force_fx, force_fy, force_fz
          - force_mx, force_my, force_mz
        """
        try:
            # 兼容字段缺失：只要能提取到6维中的一部分，也会尽可能返回可用data
            values = [
                getattr(msg, 'force_fx', None),
                getattr(msg, 'force_fy', None),
                getattr(msg, 'force_fz', None),
                getattr(msg, 'force_mx', None),
                getattr(msg, 'force_my', None),
                getattr(msg, 'force_mz', None),
            ]
            if all(v is None for v in values):
                return None
            # 将 None 替换为 0.0，保证维度固定为6（便于下游学习/对齐）
            values = [0.0 if v is None else float(v) for v in values]
            return {
                'data': np.asarray(values, dtype=np.float64),  # shape: (6,)
                'timestamp': 0.0,
                'force': np.asarray(values[:3], dtype=np.float64),
                'torque': np.asarray(values[3:], dtype=np.float64),
            }
        except Exception as e:
            log_debug(f"Sixforce处理失败: {e}")
            return None
    
    def _process_float32_multiarray(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理Float32MultiArray消息"""
        try:
            return {
                'data': np.array(msg.data, dtype=np.float32),
                'timestamp': 0.0,
                'layout': {
                    'dim': [{'label': dim.label, 'size': dim.size, 'stride': dim.stride} 
                           for dim in msg.layout.dim] if hasattr(msg, 'layout') else []
                }
            }
        except Exception as e:
            log_debug(f"Float32MultiArray处理失败: {e}")
            return None

    def _process_float32(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理Float32消息"""
        try:
            value = float(msg.data)
            return {
                'data': np.array([value], dtype=np.float32),
                'value': value,
                'timestamp': 0.0 # std_msgs通常没有header
            }
        except Exception as e:
            log_debug(f"Float32处理失败: {e}")
            return None

    def _process_float64(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理Float64消息"""
        try:
            value = float(msg.data)
            return {
                'data': np.array([value], dtype=np.float64),
                'value': value,
                'timestamp': 0.0
            }
        except Exception as e:
            log_debug(f"Float64处理失败: {e}")
            return None
    
    def _process_twist(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理Twist消息"""
        try:
            linear = [msg.linear.x, msg.linear.y, msg.linear.z]
            angular = [msg.angular.x, msg.angular.y, msg.angular.z]
            return {
                'data': np.array(linear + angular, dtype=getattr(np, config.data_type)),
                'timestamp': 0.0,
                'linear': np.array(linear, dtype=getattr(np, config.data_type)),
                'angular': np.array(angular, dtype=getattr(np, config.data_type))
            }
        except Exception as e:
            log_debug(f"Twist处理失败: {e}")
            return None
    
    def _process_twist_stamped(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理TwistStamped消息"""
        try:
            # TwistStamped包含twist字段和header字段
            twist = msg.twist
            linear = [twist.linear.x, twist.linear.y, twist.linear.z]
            angular = [twist.angular.x, twist.angular.y, twist.angular.z]
            
            # 使用header中的时间戳
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            
            return {
                'data': np.array(linear + angular, dtype=getattr(np, config.data_type)),
                'timestamp': timestamp,
                'linear': np.array(linear, dtype=getattr(np, config.data_type)),
                'angular': np.array(angular, dtype=getattr(np, config.data_type))
            }
        except Exception as e:
            log_debug(f"TwistStamped处理失败: {e}")
            return None
    
    def _process_pose_stamped(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理PoseStamped消息"""
        try:
            position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            orientation = [msg.pose.orientation.x, msg.pose.orientation.y, 
                          msg.pose.orientation.z, msg.pose.orientation.w]
            return {
                'data': np.array(position + orientation, dtype=getattr(np, config.data_type)),
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'position': np.array(position, dtype=getattr(np, config.data_type)),
                'orientation': np.array(orientation, dtype=getattr(np, config.data_type))
            }
        except Exception as e:
            log_debug(f"PoseStamped处理失败: {e}")
            return None
    
    def _process_int32(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理Int32消息"""
        try:
            value = int(msg.data)
            data_array = np.array([value], dtype=getattr(np, config.data_type, np.int32))
            return {
                'data': data_array,
                'value': value
            }
        except Exception as e:
            log_debug(f"Int32处理失败: {e}")
            return None
    
    def _process_depth_image(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理深度图像消息（16位）"""
        try:
            height, width = msg.height, msg.width
            depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape((height, width))
            if config.custom_params and config.custom_params.get('scale'):
                scale = float(config.custom_params['scale'])
                depth_array = depth_array.astype(np.float32) * scale
            return {
                'data': depth_array,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'encoding': getattr(msg, 'encoding', '16UC1')
            }
        except Exception as e:
            log_debug(f"深度图像处理失败: {e}")
            return None
    
    def _process_custom(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理自定义消息"""
        try:
            # 这里可以添加自定义处理逻辑
            # 或者调用外部处理器
            if config.custom_processor:
                allow = os.getenv("EAI_ALLOW_CUSTOM_PROCESSOR", "false").strip().lower()
                if allow in ("1", "true", "yes", "on"):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "custom_processor", config.custom_processor
                    )
                    custom_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(custom_module)
                    if hasattr(custom_module, "process_message"):
                        return custom_module.process_message(msg, config)
                else:
                    log_debug(
                        "custom_processor 已禁用（设 EAI_ALLOW_CUSTOM_PROCESSOR=true 可开启）"
                    )
            
            # 默认处理：尝试提取基本数据
            return {
                'data': np.array([], dtype=getattr(np, config.data_type)),
                'timestamp': 0.0,
                'raw_message': str(msg)
            }
        except Exception as e:
            log_debug(f"自定义消息处理失败: {e}")
            return None

class FlexibleMcapReader:
    """灵活的MCAP读取器"""
    
    def __init__(self, topic_configs: List[TopicConfig]):
        self.topic_configs = {config.topic_name: config for config in topic_configs}
        self.processor = FlexibleMcapProcessor()
        self.data = defaultdict(list)
        # 动态发现的话题配置
        self.dynamic_configs = {}
    
    def _match_topic_patterns(self, actual_topic: str) -> Optional[TopicConfig]:
        """根据实际话题名称匹配配置中的模式"""
        for config in self.topic_configs.values():
            if config.topic_patterns:
                for pattern in config.topic_patterns:
                    if self._match_topic_pattern(actual_topic, pattern):
                        # 返回原始配置，而不是创建副本
                        # 这样数据会合并到原始话题中
                        return config
        return None
    
    def _match_topic_pattern(self, actual_topic: str, pattern: str) -> bool:
        """匹配话题名称模式"""
        import re
        # 将模式转换为正则表达式
        # 支持通配符 * 和 ?
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(regex_pattern, actual_topic))
    
    def _auto_detect_message_type(self, ros_msg, config: TopicConfig) -> str:
        """自动检测消息类型"""
        if not config.auto_detect_type:
            return config.message_type
        
        # 根据消息属性判断类型
        if hasattr(ros_msg, 'data') and hasattr(ros_msg, 'format'):
            return 'compressed_image'
        elif hasattr(ros_msg, 'data') and hasattr(ros_msg, 'height') and hasattr(ros_msg, 'width'):
            return 'image'
        else:
            return config.message_type  # 回退到默认类型
    
    def process_mcap(self, mcap_file: str, monitor: 'PerformanceMonitor' = None) -> Dict[str, List[Dict[str, Any]]]:
        """处理MCAP文件，跳过损坏的记录和未配置的话题"""
        log_info(f"开始处理MCAP文件: {mcap_file}")
        
        # 初始化数据字典
        for topic_name in self.topic_configs.keys():
            self.data[topic_name] = []
        
        # 统计信息
        total_messages = 0
        processed_messages = 0
        error_count = 0
        skipped_topics = set()
        
        try:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                
                try:
                    last_log_time = time.time()
                    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                        total_messages += 1
                        
                        if total_messages % 1000 == 0:
                            current_time = time.time()
                            if current_time - last_log_time > 5.0:
                                log_info(f"处理进度: {total_messages} 条消息...", verbose_only=True)
                                last_log_time = current_time
                                gc.collect()  # 避免内存持续增长

                        topic = channel.topic
                        
                        # 首先检查精确匹配
                        if topic in self.topic_configs:
                            config = self.topic_configs[topic]
                        else:
                            # 尝试模式匹配
                            config = self._match_topic_patterns(topic)
                            if config:
                                # 记录动态发现的话题（用于日志）
                                if topic not in self.dynamic_configs:
                                    self.dynamic_configs[topic] = config
                                    log_info(f"动态发现话题: {topic} (匹配模式: {config.topic_patterns})")
                                # 使用原始配置的话题名称作为数据键
                                original_topic = config.topic_name
                                if original_topic not in self.data:
                                    self.data[original_topic] = []
                                # 将数据存储到原始话题下
                                topic = original_topic
                            else:
                                # 跳过未配置的话题
                                if topic not in skipped_topics:
                                    skipped_topics.add(topic)
                                    log_info(f"跳过未配置的话题: {topic}", verbose_only=True)
                                continue
                        try:
                            # 自动检测消息类型（如果启用）
                            detected_type = self._auto_detect_message_type(ros_msg, config)
                            if detected_type != config.message_type:
                                log_info(f"自动检测消息类型: {topic} {config.message_type} -> {detected_type}", verbose_only=True)
                                # 创建临时配置用于处理
                                temp_config = TopicConfig(
                                    topic_name=config.topic_name,
                                    message_type=detected_type,
                                    hdf5_path=config.hdf5_path,
                                    data_type=config.data_type,
                                    description=config.description,
                                    custom_processor=config.custom_processor,
                                    custom_params=config.custom_params,
                                    topic_patterns=config.topic_patterns,
                                    auto_detect_type=config.auto_detect_type,
                                    force_create_subdatasets=config.force_create_subdatasets
                                )
                                config = temp_config
                            
                            # 获取处理器：优先使用custom_processor，其次才是message_type
                            processor_key = config.custom_processor or config.message_type
                            processor = self.processor.processors.get(
                                processor_key,
                                self.processor.processors['custom']
                            )
                            
                            # 处理消息
                            msg_data = processor(ros_msg, config)
                            
                            if msg_data is not None:
                                # 使用MCAP的时间戳
                                correct_timestamp = message.log_time / 1e9
                                msg_data["timestamp"] = correct_timestamp
                                self.data[topic].append(msg_data)
                                processed_messages += 1
                                
                        except Exception as e:
                            error_count += 1
                            # 对于关键话题（如底盘），输出更详细的错误信息
                            if 'chassis' in topic.lower() or error_count <= 10:
                                log_warning(f"处理消息时出错 {topic}: {e}", verbose_only=True)
                                import traceback
                                log_debug(f"详细错误信息: {traceback.format_exc()}")
                            else:
                                log_debug(f"处理消息时出错 {topic}: {e}")
                            continue
                            

                            
                except Exception as e:
                    log_warning(f"MCAP文件读取过程中遇到错误: {e}")
                    log_warning(f"已处理 {processed_messages}/{total_messages} 条消息")
                    # 继续处理，不中断
                    
        except Exception as e:
            logger.error(f"打开MCAP文件失败: {e}")
            return {}
        
        # 记录总消息数到性能监控器
        if monitor:
            monitor.total_messages = total_messages
        
        log_info(f"MCAP文件处理完成:")
        log_info(f"  总消息数: {total_messages}", verbose_only=True)
        log_info(f"  成功处理: {processed_messages}")
        log_info(f"  处理错误: {error_count}", verbose_only=True)
        if skipped_topics:
            log_info(f"  跳过的未配置话题: {', '.join(sorted(skipped_topics))}", verbose_only=True)
        
        log_info(f"各配置话题数据量:")
        for topic, data_list in self.data.items():
            log_info(f"  {topic}: {len(data_list)} 条消息")
        
        # 检查是否有数据量为零的话题（只检查原始配置的话题，不包括动态发现的）
        original_topics = set(self.topic_configs.keys())
        zero_count_topics = [topic for topic, data_list in self.data.items() 
                           if len(data_list) == 0 and topic in original_topics]
        if zero_count_topics:
            # 改为警告而不是错误，因为我们已经支持创建空路径
            warning_msg = f"警告：以下配置的话题数据量为零（将创建空路径）: {', '.join(zero_count_topics)}"
            log_warning(warning_msg)
            log_warning(f"这可能是因为：1) 话题名称不匹配 2) 消息类型不匹配 3) 消息处理失败")
            log_warning(f"请使用 --verbose 查看详细日志，或使用 query_ros2_topics.py 查询实际话题")
            # 不再返回空字典，而是继续处理（会创建空路径）
        
        return dict(self.data)
    
    

    def align_data_backfill_on_grid(self, data: Dict[str, List[Dict[str, Any]]], 
                                    alignment_config: AlignmentConfig) -> Tuple[
        Dict[str, List[Dict[str, Any]]],
        Dict[str, List[Dict[str, Any]]],
        Dict[str, int],
        float,
        Dict[str, List[Dict[str, Any]]],
    ]:
        """等间隔网格回填对齐（backfill_on_grid）。"""
        log_info("开始 backfill_on_grid 对齐...")
        topics_with_data = {t: msgs for t, msgs in data.items() if len(msgs) > 0}
        if not topics_with_data:
            logger.error("没有任何可用数据用于对齐")
            ws = _alignment_window_size(alignment_config)
            return {}, {}, {}, ws, {}

        topic_starts = {t: min(m['timestamp'] for m in msgs) for t, msgs in topics_with_data.items()}
        topic_ends = {t: max(m['timestamp'] for m in msgs) for t, msgs in topics_with_data.items()}
        grid_start = max(topic_starts.values())
        grid_end = max(topic_ends.values())
        if grid_end <= grid_start:
            log_warning("网格时间范围无效，无法进行对齐", verbose_only=True)
            ws = _alignment_window_size(alignment_config)
            return {}, {}, {}, ws, {}

        fps = alignment_config.grid_fps if alignment_config.grid_fps and alignment_config.grid_fps > 0 else 15.0
        period = 1.0 / fps
        grid_times: List[float] = []
        t = grid_start
        while t <= grid_end + 1e-9:
            grid_times.append(t)
            t += period
        log_info(f"backfill_on_grid 网格步数: {len(grid_times)}, 频率: {fps}Hz, 范围: [{grid_start:.6f}, {grid_end:.6f}] (使用最晚结束时间)")

        # 预计算每个话题的时间戳数组
        topic_ts: Dict[str, np.ndarray] = {}
        for topic, msgs in topics_with_data.items():
            topic_ts[topic] = np.array([m['timestamp'] for m in msgs], dtype=np.float64)

        # Cython 下勿写 Dict[...] = defaultdict(...)（会触发类型检查）；返回时再 dict()
        aligned_data = defaultdict(list)
        alignment_details = defaultdict(list)
        dropped_counts = defaultdict(int)
        warning_stats = defaultdict(list)
        for target_t in grid_times:
            row: Dict[str, Dict[str, Any]] = {}
            all_found = True
            for topic, msgs in topics_with_data.items():
                ts = topic_ts[topic]
                # 在 [target_t - period, target_t] 内找"最新"数据，若没有则选择 <= target_t 的最近历史数据
                start_t = target_t - period
                right = np.searchsorted(ts, target_t, side='right')
                left = np.searchsorted(ts, start_t, side='left')
                chosen_idx = None
                over_period = False
                if right - left > 0:
                    chosen_idx = right - 1
                else:
                    # 选取不大于 target_t 的最近历史数据（无限回填）
                    if right > 0:
                        chosen_idx = right - 1
                        over_period = True
                    else:
                        # 没有 <= target_t 的数据（全部都在更早或该话题开始晚于网格起点）
                        if alignment_config.expand_start and len(ts) > 0:
                            # 允许越过网格起点，选择该话题的最早数据
                            chosen_idx = 0
                            over_period = True
                        else:
                            all_found = False
                            dropped_counts[topic] += 1
                            warning_msg = '无可用于回填的历史数据（受限于网格起点）'
                            log_warning(f"❌ 话题 {topic} {warning_msg}: 时间步={target_t:.6f}s", verbose_only=True)
                            alignment_details[topic].append({
                                'target_time': target_t,
                                'selected_time': None,
                                'time_diff': None,
                                'abs_time_diff': None,
                                'warning': warning_msg
                            })
                            continue

                if chosen_idx is not None:
                    row[topic] = msgs[chosen_idx]
                    sel_t = float(topic_ts[topic][chosen_idx])
                    detail = {
                        'target_time': target_t,
                        'selected_time': sel_t,
                        'time_diff': sel_t - target_t,
                        'abs_time_diff': abs(sel_t - target_t),
                    }
                    if over_period:
                        # 超过一个周期或越过网格起点才找到，记录为警告但不影响对齐
                        if sel_t < target_t - period:
                            detail['warning'] = 'Backfill beyond one period'
                            log_warning(f"⚠️  Topic {topic} backfill beyond one period: target_time={target_t:.6f}s, selected_time={sel_t:.6f}s, diff={sel_t-target_t:+.6f}s", verbose_only=True)
                            # 记录到报警统计中
                            warning_stats[topic].append({
                                'warning': 'Backfill beyond one period',
                                'target_time': target_t,
                                'selected_time': sel_t,
                                'time_diff': sel_t - target_t,
                                'period': period
                            })
                        elif sel_t < grid_start:
                            detail['warning'] = 'Backfill beyond grid start'
                            log_warning(f"⚠️  Topic {topic} backfill beyond grid start: target_time={target_t:.6f}s, selected_time={sel_t:.6f}s, grid_start={grid_start:.6f}s", verbose_only=True)
                            # 记录到报警统计中
                            warning_stats[topic].append({
                                'warning': 'Backfill beyond grid start',
                                'target_time': target_t,
                                'selected_time': sel_t,
                                'time_diff': sel_t - target_t,
                                'grid_start': grid_start
                            })
                    alignment_details[topic].append(detail)

            if all_found and row:
                for topic, msg in row.items():
                    aligned_data[topic].append(msg)
            else:
                dropped_counts['all_topics'] += 1

        log_info("backfill_on_grid 对齐完成")
        window_size = _alignment_window_size(alignment_config)
        return (
            dict(aligned_data),
            dict(alignment_details),
            dict(dropped_counts),
            window_size,
            dict(warning_stats),
        )
    
    

class FlexibleHdf5Converter:
    """灵活的HDF5转换器"""
    
    def __init__(self, topic_configs: List[TopicConfig], alignment_config: AlignmentConfig, 
                 data_merging_config: Dict[str, Any] = None):
        self.topic_configs = {config.topic_name: config for config in topic_configs}
        self.alignment_config = alignment_config
        self.data_merging_config = data_merging_config or {}
    
    def _write_chunked_dataset(self, hdf5_group, path, data_list, key, dtype):
        """
        分块写入数据集以降低内存占用。
        data_list: 消息列表
        key: 从消息中提取数据的键（如 'data', 'depth'）
        dtype: 目标数据类型
        """
        # 筛选有效消息
        valid_msgs = [msg for msg in data_list if msg.get(key) is not None]
        if not valid_msgs:
            if 'image' in path or 'compress' in path:
                 self._create_empty_image_dataset(hdf5_group, path)
            else:
                 self._create_empty_dataset(hdf5_group, path, dtype)
            return

        count = len(valid_msgs)
        first_val = valid_msgs[0][key]
        
        # 获取单个元素的形状
        if isinstance(first_val, np.ndarray):
             item_shape = first_val.shape
        else:
             item_shape = ()
        
        full_shape = (count,) + item_shape
        
        # 删除旧数据集（如果存在）
        if path in hdf5_group:
            del hdf5_group[path]
            
        # 创建数据集 - 不启用压缩
        dset = hdf5_group.create_dataset(
            path, 
            shape=full_shape, 
            dtype=dtype
        )
        
        # 逐个写入
        for i, msg in enumerate(valid_msgs):
            val = msg[key]
            # 处理类型转换
            if hasattr(val, 'dtype') and val.dtype != dtype:
                 dset[i] = val.astype(dtype)
            else:
                 dset[i] = val
        
        log_info(f"分块保存数据: {path} {full_shape}", verbose_only=True)

    def convert_to_hdf5(self, aligned_data: Dict[str, List[Dict[str, Any]]], 
                       output_path: str, alignment_details: Dict = None, 
                       dropped_counts: Dict = None, window_size: float = None, 
                       warning_stats: Dict = None, original_data: Dict = None, original_mcap_file: str = None) -> bool:
        """转换对齐后的数据到HDF5"""
        try:
            log_info(f"开始转换到HDF5: {output_path}")
            
            # 应用目标时长限制（如果启用）
            if (
                self.alignment_config.target_duration is not None
                and self.alignment_config.target_duration > 0
            ):
                aligned_data = self._apply_fixed_duration(aligned_data)
            
            # 如果文件已存在，先删除
            if os.path.exists(output_path):
                log_info(f"删除已存在的HDF5文件: {output_path}", verbose_only=True)
                os.remove(output_path)
            
            with h5py.File(output_path, 'w') as f:
                # 转换每个话题的数据
                for topic_name, data_list in aligned_data.items():
                    config = self.topic_configs[topic_name]
                    hdf5_path = config.hdf5_path
                    
                    if len(data_list) == 0:
                        # 即使没有数据，也要为所有类型的话题创建路径
                        if config.message_type in ['compressed_image', 'image']:
                            origin_path = f"{hdf5_path}/color/origin"
                            compress_path = f"{hdf5_path}/color/compress"
                            depth_path = f"{hdf5_path}/depth"
                            pointcloud_path = f"{hdf5_path}/pointcloud"
                            self._ensure_image_group_no_overwrite(f, origin_path)
                            self._ensure_image_group_no_overwrite(f, compress_path)
                            self._ensure_empty_dataset_no_overwrite(f, depth_path, np.float32)
                            self._ensure_empty_dataset_no_overwrite(f, pointcloud_path, np.float32)
                            log_info(f"创建空图像路径（无数据）: {origin_path}, {compress_path}, {depth_path}, {pointcloud_path}", verbose_only=True)
                        elif config.message_type == 'joint_state' or config.message_type == 'twist_stamped':
                            # 关节状态：创建 qpos, effort, vel
                            qpos_path = f"{hdf5_path}/qpos"
                            effort_path = f"{hdf5_path}/effort"
                            vel_path = f"{hdf5_path}/vel"
                            self._create_empty_dataset(f, qpos_path, getattr(np, config.data_type))
                            self._create_empty_dataset(f, effort_path, getattr(np, config.data_type))
                            self._create_empty_dataset(f, vel_path, getattr(np, config.data_type))
                            log_info(f"创建空关节状态路径（无数据）: {qpos_path}, {effort_path}, {vel_path}", verbose_only=True)
                        elif config.message_type == 'pose_stamped':
                            # 位姿数据：创建 data
                            data_path = f"{hdf5_path}/data"
                            self._create_empty_dataset(f, data_path, getattr(np, config.data_type))
                            log_info(f"创建空位姿路径（无数据）: {data_path}", verbose_only=True)
                        else:
                            # 其他类型（如夹爪）：创建 data
                            data_path = f"{hdf5_path}/data"
                            self._create_empty_dataset(f, data_path, getattr(np, config.data_type))
                            log_info(f"创建空数据路径（无数据）: {data_path}", verbose_only=True)
                        continue
                    
                    # 提取时间戳数据
                    num_timesteps = len(data_list)
                    timestamps = np.array([msg['timestamp'] for msg in data_list], dtype=np.float64)
                    
                    # 提取数据
                    if config.message_type in ['compressed_image', 'image']:
                        # 图像数据 - 按照新结构保存
                        # 新结构：需要创建 color/origin, color/compress, depth, pointcloud 四个路径
                        # hdf5_path 应该是 /images/cam_head 这样的路径
                        origin_path = f"{hdf5_path}/color/origin"
                        compress_path = f"{hdf5_path}/color/compress"
                        depth_path = f"{hdf5_path}/depth"
                        pointcloud_path = f"{hdf5_path}/pointcloud"
                        
                        # 关键修复：
                        # 1) 彩色 topic（compressed_image）不应因为没有 depth 而把已有 depth 数据“清空”
                        # 2) 深度 topic（image，但 msg 里只有 depth 字段）不应因为没有 data 而把已有彩色数据“清空”
                        has_color_data = any(msg.get('data') is not None for msg in data_list)
                        has_depth_data = any(msg.get('depth') is not None for msg in data_list)
                        has_pointcloud_data = any(msg.get('pointcloud') is not None for msg in data_list)

                        if has_color_data:
                            if config.message_type == 'compressed_image':
                                self._write_chunked_dataset(f, compress_path, data_list, 'data', np.uint8)
                                self._ensure_image_group_no_overwrite(f, origin_path)
                            else:
                                self._write_chunked_dataset(f, origin_path, data_list, 'data', np.uint8)
                                self._ensure_image_group_no_overwrite(f, compress_path)
                        else:
                            # 深度 topic：不写 color；只在路径缺失时确保 group 存在，避免覆盖已有彩色数据集
                            self._ensure_image_group_no_overwrite(f, origin_path)
                            self._ensure_image_group_no_overwrite(f, compress_path)

                        if has_depth_data:
                            self._write_chunked_dataset(f, depth_path, data_list, 'depth', np.float32)

                        if has_pointcloud_data:
                            self._write_chunked_dataset(f, pointcloud_path, data_list, 'pointcloud', np.float32)
                    
                    elif config.message_type in ['joint_state', 'twist_stamped']:
                        # 关节状态或TwistStamped数据 - 根据配置处理子数据集
                        dtype_np = getattr(np, config.data_type)
                        force_create = getattr(config, 'force_create_subdatasets', False)

                        def pad_or_truncate(arr: Any, expected_dim: int, use_nan: bool = True) -> np.ndarray:
                            """根据期望维度调整数组长度，不足填充、超出截断。"""
                            array = np.asarray(arr, dtype=dtype_np).flatten()
                            if expected_dim <= 0:
                                return array
                            if array.shape[0] < expected_dim:
                                fill_value = np.nan if (force_create and use_nan) else 0.0
                                pad = np.full(expected_dim - array.shape[0], fill_value, dtype=dtype_np)
                                array = np.concatenate([array, pad])
                            elif array.shape[0] > expected_dim:
                                array = array[:expected_dim]
                            return array

                        def write_subdataset(key: str, suffix: str, use_actual_dim: bool = True):
                            """写入子数据集，使用实际维度（不强制统一）"""
                            sub_path = f"{hdf5_path}/{suffix}"
                            
                            rows: List[np.ndarray] = []
                            has_value = False
                            actual_dims = []
                            
                            for msg in data_list:
                                value = msg.get(key)
                                if value is not None:
                                    has_value = True
                                    arr = np.asarray(value, dtype=dtype_np).flatten()
                                    rows.append(arr)
                                    actual_dims.append(len(arr))
                                elif force_create:
                                    # 如果启用force_create，需要确定维度
                                    # 使用该字段的最大维度
                                    max_dim = max(actual_dims) if actual_dims else 0
                                    if max_dim > 0:
                                        rows.append(np.full(max_dim, np.nan, dtype=dtype_np))
                                    # 如果无法确定维度，跳过这条消息
                            
                            if not has_value:
                                self._create_empty_dataset(f, sub_path, dtype_np)
                                return
                            
                            if not rows:
                                self._create_empty_dataset(f, sub_path, dtype_np)
                                return
                            
                            # 确定目标维度：使用该字段的最大维度
                            target_dim = max(actual_dims) if actual_dims else 0
                            if target_dim == 0:
                                self._create_empty_dataset(f, sub_path, dtype_np)
                                return
                            
                            # 统一维度（填充或截取到最大维度）
                            for i, row in enumerate(rows):
                                if len(row) < target_dim:
                                    padding = np.zeros(target_dim - len(row), dtype=dtype_np)
                                    rows[i] = np.concatenate([row, padding])
                                elif len(row) > target_dim:
                                    rows[i] = row[:target_dim]
                            
                            # 如果行数不足，补充NaN行
                            if len(rows) < num_timesteps and force_create:
                                deficit = num_timesteps - len(rows)
                                if deficit > 0:
                                    rows.extend([np.full(target_dim, np.nan, dtype=dtype_np) for _ in range(deficit)])
                            
                            stacked = np.stack(rows, axis=0)
                            self._create_or_replace_dataset(f, sub_path, stacked, dtype_np)

                        if config.message_type == 'joint_state':
                            # 主数据：qpos - 使用position的实际维度
                            position_rows = []
                            for msg in data_list:
                                if msg.get('data') is not None:
                                    position_rows.append(np.asarray(msg['data'], dtype=dtype_np).flatten())
                            
                            if position_rows:
                                # 使用position的最大维度
                                max_pos_dim = max(len(row) for row in position_rows)
                                # 统一维度（填充或截取）
                                for i, row in enumerate(position_rows):
                                    if len(row) < max_pos_dim:
                                        padding = np.zeros(max_pos_dim - len(row), dtype=dtype_np)
                                        position_rows[i] = np.concatenate([row, padding])
                                    elif len(row) > max_pos_dim:
                                        position_rows[i] = row[:max_pos_dim]
                                positions = np.stack(position_rows, axis=0)
                                qpos_path = f"{hdf5_path}/qpos"
                                self._create_or_replace_dataset(f, qpos_path, positions, dtype_np)
                                log_info(f"保存关节状态: qpos={qpos_path} {positions.shape}, 时间戳: {timestamps.shape}", verbose_only=True)
                            else:
                                qpos_path = f"{hdf5_path}/qpos"
                                self._create_empty_dataset(f, qpos_path, dtype_np)
                                log_info(f"保存关节状态: qpos={qpos_path} (空), 时间戳: {timestamps.shape}", verbose_only=True)
                            
                            # velocity 和 effort 分别处理，使用各自的维度
                            write_subdataset('velocity', 'vel', use_actual_dim=True)
                            write_subdataset('effort', 'effort', use_actual_dim=True)

                        else:
                            # TwistStamped 数据
                            twist_rows = [np.asarray(msg['data'], dtype=dtype_np).flatten() for msg in data_list]
                            main_dim = max((row.shape[0] for row in twist_rows), default=0)
                            if main_dim > 0:
                                twist_rows = [pad_or_truncate(row, main_dim, use_nan=False) for row in twist_rows]
                                twist_array = np.stack(twist_rows, axis=0)
                                qpos_path = f"{hdf5_path}/qpos"
                                self._create_or_replace_dataset(f, qpos_path, twist_array, dtype_np)
                            else:
                                qpos_path = f"{hdf5_path}/qpos"
                                self._create_empty_dataset(f, qpos_path, dtype_np)

                            # linear 和 angular 分别处理，使用各自的维度（都是3维）
                            write_subdataset('linear', 'linear', use_actual_dim=True)
                            write_subdataset('angular', 'angular', use_actual_dim=True)

                            log_info(f"保存TwistStamped数据: {qpos_path}, 时间戳: {timestamps.shape}", verbose_only=True)
                    
                    elif config.message_type == 'sixforce' or config.custom_processor == 'sixforce':
                        # Sixforce：显式拆分 force / torque 子数据集
                        dtype_np = getattr(np, config.data_type)
                        data_path = f"{hdf5_path}/data"
                        force_path = f"{hdf5_path}/force"
                        torque_path = f"{hdf5_path}/torque"

                        data_rows = [msg.get('data') for msg in data_list if msg.get('data') is not None]
                        if data_rows:
                            data_array = np.stack([np.asarray(v, dtype=dtype_np).flatten() for v in data_rows], axis=0)
                            self._create_or_replace_dataset(f, data_path, data_array, dtype_np)
                        else:
                            self._create_empty_dataset(f, data_path, dtype_np)

                        force_rows = [msg.get('force') for msg in data_list if msg.get('force') is not None]
                        if force_rows:
                            force_array = np.stack([np.asarray(v, dtype=dtype_np).flatten() for v in force_rows], axis=0)
                            self._create_or_replace_dataset(f, force_path, force_array, dtype_np)
                        else:
                            self._create_empty_dataset(f, force_path, dtype_np)

                        torque_rows = [msg.get('torque') for msg in data_list if msg.get('torque') is not None]
                        if torque_rows:
                            torque_array = np.stack([np.asarray(v, dtype=dtype_np).flatten() for v in torque_rows], axis=0)
                            self._create_or_replace_dataset(f, torque_path, torque_array, dtype_np)
                        else:
                            self._create_empty_dataset(f, torque_path, dtype_np)
                    
                    elif config.message_type == 'pose_stamped':
                        # 位姿数据 - 按照新结构保存为 data
                        poses = np.array([msg['data'] for msg in data_list])
                        # 新结构：hdf5_path 应该是 /observations/arm_endpose_left_state 这样的路径
                        # 需要保存为 /observations/arm_endpose_left_state/data
                        data_path = f"{hdf5_path}/data"
                        self._create_or_replace_dataset(f, data_path, poses, getattr(np, config.data_type))
                        log_info(f"保存位姿数据: {data_path} {poses.shape}, 时间戳: {timestamps.shape}", verbose_only=True)
                    
                    else:
                        # 其他数据类型（如夹爪数据）- 按照新结构保存为 data
                        data_path = f"{hdf5_path}/data"
                        data_arrays = [msg['data'] for msg in data_list]
                        if data_arrays and len(data_arrays[0]) > 0:
                            combined_data = np.stack(data_arrays)
                            # 新结构：对于夹爪等数据，hdf5_path 应该是 /observations/gripper_left_state
                            # 需要保存为 /observations/gripper_left_state/data
                            self._create_or_replace_dataset(f, data_path, combined_data, getattr(np, config.data_type))
                            log_info(f"保存数据: {data_path} {combined_data.shape}, 时间戳: {timestamps.shape}", verbose_only=True)
                        else:
                            # 即使没有数据，也创建空路径
                            self._create_empty_dataset(f, data_path, getattr(np, config.data_type))
                            log_info(f"创建空数据路径: {data_path}", verbose_only=True)
                
                # 为所有配置的话题创建路径（即使没有匹配到数据）
                for topic_name, config in self.topic_configs.items():
                    if topic_name not in aligned_data:
                        # 没有匹配到数据，根据类型创建相应的空路径
                        if config.message_type in ['compressed_image', 'image']:
                            # 图像数据：创建 origin, compress, depth, pointcloud 四个路径
                            origin_path = f"{config.hdf5_path}/color/origin"
                            compress_path = f"{config.hdf5_path}/color/compress"
                            depth_path = f"{config.hdf5_path}/depth"
                            pointcloud_path = f"{config.hdf5_path}/pointcloud"
                            self._ensure_image_group_no_overwrite(f, origin_path)
                            self._ensure_image_group_no_overwrite(f, compress_path)
                            self._ensure_empty_dataset_no_overwrite(f, depth_path, np.float32)
                            self._ensure_empty_dataset_no_overwrite(f, pointcloud_path, np.float32)
                            log_info(f"创建空图像路径（未匹配到数据）: {origin_path}, {compress_path}, {depth_path}, {pointcloud_path}", verbose_only=True)
                        elif config.message_type == 'joint_state' or config.message_type == 'twist_stamped':
                            # 关节状态：创建 qpos, effort, vel 三个路径
                            qpos_path = f"{config.hdf5_path}/qpos"
                            effort_path = f"{config.hdf5_path}/effort"
                            vel_path = f"{config.hdf5_path}/vel"
                            self._create_empty_dataset(f, qpos_path, getattr(np, config.data_type))
                            self._create_empty_dataset(f, effort_path, getattr(np, config.data_type))
                            self._create_empty_dataset(f, vel_path, getattr(np, config.data_type))
                            log_info(f"创建空关节状态路径（未匹配到数据）: {qpos_path}, {effort_path}, {vel_path}", verbose_only=True)
                        elif config.message_type == 'sixforce' or config.custom_processor == 'sixforce':
                            # Sixforce：创建 data / force / torque 三个数据集（空）
                            data_path = f"{config.hdf5_path}/data"
                            force_path = f"{config.hdf5_path}/force"
                            torque_path = f"{config.hdf5_path}/torque"
                            self._ensure_empty_dataset_no_overwrite(f, data_path, getattr(np, config.data_type))
                            self._ensure_empty_dataset_no_overwrite(f, force_path, getattr(np, config.data_type))
                            self._ensure_empty_dataset_no_overwrite(f, torque_path, getattr(np, config.data_type))
                        elif config.message_type == 'pose_stamped':
                            # 位姿数据：创建 data 路径
                            data_path = f"{config.hdf5_path}/data"
                            self._create_empty_dataset(f, data_path, getattr(np, config.data_type))
                            log_info(f"创建空位姿路径（未匹配到数据）: {data_path}", verbose_only=True)
                        else:
                            # 其他类型（如夹爪）：创建 data 路径
                            data_path = f"{config.hdf5_path}/data"
                            self._create_empty_dataset(f, data_path, getattr(np, config.data_type))
                            log_info(f"创建空数据路径（未匹配到数据）: {data_path}", verbose_only=True)
                
                # 应用后处理
                self._apply_post_processing(f, aligned_data)
                
                # 保存配置信息
                self._save_config_info(f)
                
                # 保存报警统计到HDF5文件中
                if warning_stats is not None:
                    self._save_warning_stats_to_hdf5(f, warning_stats, original_data, original_mcap_file)
            
            log_info(f"HDF5文件保存成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"HDF5转换失败: {e}")
            return False
    
    def _apply_fixed_duration(self, aligned_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """应用固定时长限制：使用最晚的时间戳作为起始点，截取或填充数据到指定时长"""
        target_duration = self.alignment_config.target_duration
        if target_duration is None or target_duration <= 0:
            return aligned_data
        
        # 找到所有话题中最晚的起始时间戳作为起始时间
        start_times = []
        for topic, msgs in aligned_data.items():
            if msgs:
                start_times.append(msgs[0]['timestamp'])
        
        if not start_times:
            return aligned_data
            
        start_time = max(start_times)
        end_time = start_time + target_duration
        
        log_info(f"应用固定时长限制: {target_duration}s, 时间范围: [{start_time:.3f}, {end_time:.3f}]")
        
        result = {}
        fps = self.alignment_config.grid_fps if self.alignment_config.grid_fps and self.alignment_config.grid_fps > 0 else 15.0
        period = 1.0 / fps
        expected_count = int(np.ceil(target_duration * fps))
        
        for topic, msgs in aligned_data.items():
            # 截取在范围内的消息
            filtered = [m for m in msgs if start_time - 1e-6 <= m['timestamp'] <= end_time + 1e-6]
            
            # 如果数量不足，且配置为force_create，需要填充
            # 但这里已经是 aligned_data，理论上是对齐过的
            # 如果截取后数量少于 expected_count，说明原始数据就不够长
            # 这里简单返回截取后的数据
            result[topic] = filtered
            
        return result


    
    def _apply_post_processing(self, hdf5_file, aligned_data: Dict[str, List[Dict[str, Any]]]):
        """应用后处理"""
        # 根据配置文件进行数据合并处理
        self._merge_data_according_to_config(hdf5_file, aligned_data)
        
        if self.alignment_config.relative_start:
            # 相对起始点处理
            for topic_name, data_list in aligned_data.items():
                if len(data_list) == 0:
                    continue
                
                config = self.topic_configs[topic_name]
                hdf5_path = config.hdf5_path
                
                if hdf5_path in hdf5_file:
                    dataset = hdf5_file[hdf5_path]
                    if dataset.size > 0:
                        # 减去第一帧的值
                        first_frame = dataset[0]
                        dataset[:] = dataset[:] - first_frame
                        log_info(f"应用相对起始点处理: {hdf5_path}", verbose_only=True)
        
        if self.alignment_config.delta_action:
            # 增量动作处理
            for topic_name, data_list in aligned_data.items():
                if len(data_list) == 0:
                    continue
                
                config = self.topic_configs[topic_name]
                hdf5_path = config.hdf5_path
                
                if hdf5_path in hdf5_file and 'action' in hdf5_path.lower():
                    dataset = hdf5_file[hdf5_path]
                    if dataset.shape[0] > 1:
                        # 计算增量
                        delta_data = dataset[1:] - dataset[:-1]
                        # 更新数据集
                        del hdf5_file[hdf5_path]
                        hdf5_file.create_dataset(hdf5_path, data=delta_data, 
                                               dtype=dataset.dtype)
                        log_info(f"应用增量动作处理: {hdf5_path}", verbose_only=True)
    
    def _merge_data_according_to_config(self, hdf5_file, aligned_data: Dict[str, List[Dict[str, Any]]]):
        """根据配置文件合并数据"""
        try:
            # 处理数据合并配置
            for merge_name, merge_config in self.data_merging_config.items():
                target_path = merge_config.get('target_path')
                source_topics = merge_config.get('source_topics', [])
                joint_counts = merge_config.get('joint_counts', [])
                
                if not target_path or not source_topics:
                    continue
                
                # 收集源数据
                source_data = []
                for i, topic_name in enumerate(source_topics):
                    # 查找对应的HDF5路径
                    topic_config = self.topic_configs.get(topic_name)
                    if not topic_config:
                        log_warning(f"未找到话题配置: {topic_name}", verbose_only=True)
                        continue
                    
                    source_path = topic_config.hdf5_path
                    if source_path in hdf5_file:
                        data = hdf5_file[source_path][:]
                        source_data.append(data)
                        log_info(f"收集源数据 {topic_name}: {data.shape}", verbose_only=True)
                
                # 合并数据
                if len(source_data) > 1:
                    merged_data = np.concatenate(source_data, axis=1)
                    self._create_or_replace_dataset(hdf5_file, target_path, merged_data, np.float32)
                    
                    # 合并时间戳数据（使用第一个话题的时间戳作为参考）
                    first_topic = source_topics[0]
                    first_topic_config = self.topic_configs.get(first_topic)
                    if first_topic_config:
                        first_timestamps_path = f"{first_topic_config.hdf5_path}_timestamps"
                        if first_timestamps_path in hdf5_file:
                            timestamps = hdf5_file[first_timestamps_path][:]
                            self._create_or_replace_dataset(hdf5_file, f"{target_path}_timestamps", timestamps, np.float64)
                            log_info(f"合并时间戳到 {target_path}_timestamps: {timestamps.shape}", verbose_only=True)
                    
                    log_info(f"合并数据到 {target_path}: {merged_data.shape}")
                    
                    # 合并速度和力矩数据
                    self._merge_velocity_effort_data(hdf5_file, source_topics, target_path, merge_config)
                    
        except Exception as e:
            log_warning(f"数据合并处理失败: {e}", verbose_only=True)
    
    def _merge_velocity_effort_data(self, hdf5_file, source_topics: List[str], target_path: str, merge_config: Dict[str, Any]):
        """合并速度和力矩数据 - 适配新的配置文件结构"""
        try:
            # 合并速度数据
            velocity_data = []
            effort_data = []
            joint_counts = merge_config.get('joint_counts', [])
            
            for i, topic_name in enumerate(source_topics):
                topic_config = self.topic_configs.get(topic_name)
                if not topic_config:
                    continue
                
                # 获取源路径的基础部分（去掉/position后缀）
                source_base_path = topic_config.hdf5_path
                if source_base_path.endswith('/position'):
                    source_base_path = source_base_path[:-9]  # 去掉"/position"
                elif source_base_path.endswith('_position'):
                    source_base_path = source_base_path[:-9]  # 去掉"_position"
                
                # 收集速度数据
                vel_path = f"{source_base_path}/velocity"
                if vel_path in hdf5_file:
                    vel_data = hdf5_file[vel_path][:]
                    # 如果指定了关节数量，则截取前n个关节
                    if i < len(joint_counts) and joint_counts[i] > 0:
                        vel_data = vel_data[:, :joint_counts[i]]
                    velocity_data.append(vel_data)
                    log_info(f"收集速度数据 {topic_name}: {vel_path} -> {vel_data.shape}", verbose_only=True)
                else:
                    # 检查是否有其他可能的速度数据路径
                    alt_vel_paths = [
                        f"{topic_config.hdf5_path.replace('/position', '/velocity')}",
                        f"{topic_config.hdf5_path.replace('_position', '_velocity')}",
                        f"{topic_config.hdf5_path}_velocity"
                    ]
                    
                    found_vel = False
                    for alt_path in alt_vel_paths:
                        if alt_path in hdf5_file:
                            vel_data = hdf5_file[alt_path][:]
                            if i < len(joint_counts) and joint_counts[i] > 0:
                                vel_data = vel_data[:, :joint_counts[i]]
                            velocity_data.append(vel_data)
                            log_info(f"收集速度数据 {topic_name}: {alt_path} -> {vel_data.shape}", verbose_only=True)
                            found_vel = True
                            break
                    
                    if not found_vel:
                        log_debug(f"未找到速度数据: {vel_path} (尝试了备用路径: {alt_vel_paths})")
                
                # 收集力矩数据
                eff_path = f"{source_base_path}/effort"
                if eff_path in hdf5_file:
                    eff_data = hdf5_file[eff_path][:]
                    # 如果指定了关节数量，则截取前n个关节
                    if i < len(joint_counts) and joint_counts[i] > 0:
                        eff_data = eff_data[:, :joint_counts[i]]
                    effort_data.append(eff_data)
                    log_info(f"收集力矩数据 {topic_name}: {eff_path} -> {eff_data.shape}", verbose_only=True)
                else:
                    # 检查是否有其他可能的力矩数据路径
                    alt_eff_paths = [
                        f"{topic_config.hdf5_path.replace('/position', '/effort')}",
                        f"{topic_config.hdf5_path.replace('_position', '_effort')}",
                        f"{topic_config.hdf5_path}_effort"
                    ]
                    
                    found_eff = False
                    for alt_path in alt_eff_paths:
                        if alt_path in hdf5_file:
                            eff_data = hdf5_file[alt_path][:]
                            if i < len(joint_counts) and joint_counts[i] > 0:
                                eff_data = eff_data[:, :joint_counts[i]]
                            effort_data.append(eff_data)
                            log_info(f"收集力矩数据 {topic_name}: {alt_path} -> {eff_data.shape}", verbose_only=True)
                            found_eff = True
                            break
                    
                    if not found_eff:
                        log_debug(f"未找到力矩数据: {eff_path} (尝试了备用路径: {alt_eff_paths})")
            
            # 合并速度数据
            if velocity_data and len(velocity_data) > 0:
                merged_velocity = np.concatenate(velocity_data, axis=1)
                velocity_target_path = target_path.replace('/position', '/velocity')
                self._create_or_replace_dataset(hdf5_file, velocity_target_path, merged_velocity, np.float32)
                log_info(f"合并速度数据到 {velocity_target_path}: {merged_velocity.shape}", verbose_only=True)
            else:
                log_warning(f"没有找到任何速度数据可用于合并到 {target_path.replace('/position', '/velocity')}", verbose_only=True)
            
            # 合并力矩数据
            if effort_data and len(effort_data) > 0:
                merged_effort = np.concatenate(effort_data, axis=1)
                effort_target_path = target_path.replace('/position', '/effort')
                self._create_or_replace_dataset(hdf5_file, effort_target_path, merged_effort, np.float32)
                log_info(f"合并力矩数据到 {effort_target_path}: {merged_effort.shape}", verbose_only=True)
            else:
                log_warning(f"没有找到任何力矩数据可用于合并到 {target_path.replace('/position', '/effort')}", verbose_only=True)
                
        except Exception as e:
            log_warning(f"合并速度和力矩数据失败: {e}", verbose_only=True)
            import traceback
            log_debug(traceback.format_exc())

    def _create_empty_image_dataset(self, h5file: h5py.File, path: str):
        """
        为图像路径创建一个空的 group。
        这样可以保证路径存在，同时不占用额外空间。
        """
        try:
            parent = '/'.join(path.split('/')[:-1])
            if parent and parent not in h5file:
                h5file.require_group(parent)

            if path in h5file:
                obj = h5file[path]
                if isinstance(obj, h5py.Dataset):
                    del h5file[path]
                    h5file.create_group(path)
                # 如果已经是group则无需重复创建
            else:
                h5file.create_group(path)
            log_debug(f"创建/保持空图像路径: {path}")
        except Exception as e:
            log_warning(f"创建空图像路径失败 {path}: {e}", verbose_only=True)

    def _ensure_image_group_no_overwrite(self, h5file: h5py.File, path: str):
        """
        仅当路径不存在时创建空 group；不会删除已存在的数据集/组。
        用于避免“后处理的某个 topic 把彩色图像/深度图像覆盖掉”的问题。
        """
        try:
            parent = '/'.join(path.split('/')[:-1])
            if parent and parent not in h5file:
                h5file.require_group(parent)
            if path in h5file:
                return
            h5file.create_group(path)
        except Exception as e:
            log_warning(f"确保空图像 group 失败 {path}: {e}", verbose_only=True)

    def _ensure_empty_dataset_no_overwrite(self, h5file: h5py.File, path: str, dtype=np.float32):
        """
        仅当路径不存在时创建 shape=(0,) 的空数据集；不会覆盖已存在的数据。
        """
        try:
            parent = '/'.join(path.split('/')[:-1])
            if parent and parent not in h5file:
                h5file.require_group(parent)
            if path in h5file:
                return
            h5file.create_dataset(path, shape=(0,), dtype=dtype)
        except Exception as e:
            log_warning(f"确保空数据集失败 {path}: {e}", verbose_only=True)
    
    def _create_empty_dataset(self, h5file: h5py.File, path: str, dtype=np.float32):
        """创建 shape=(0,) 的空数据集，确保路径存在但无实数据。"""
        try:
            parent = '/'.join(path.split('/')[:-1])
            if parent and parent not in h5file:
                h5file.require_group(parent)
            
            if path in h5file:
                del h5file[path]
            
            h5file.create_dataset(path, shape=(0,), dtype=dtype)
            log_debug(f"创建空数据集: {path}")
        except Exception as e:
            log_warning(f"创建空数据集失败 {path}: {e}", verbose_only=True)

    def _create_or_replace_dataset(self, h5file: h5py.File, path: str, data: Any, dtype: Any):
        """若数据集存在则删除后重建，避免Dataset already exists错误。"""
        try:
            # 确保父组存在
            parent = '/'.join(path.split('/')[:-1])
            if parent and parent not in h5file:
                h5file.require_group(parent)
            
            # 删除已存在的数据集或组
            if path in h5file:
                log_debug(f"删除已存在的数据集: {path}")
                del h5file[path]
            
            # 创建新数据集
            shape = data.shape
            if len(shape) >= 2 and shape[0] > 10:
                h5file.create_dataset(path, data=data, dtype=dtype)
            else:
                h5file.create_dataset(path, data=data, dtype=dtype)
            log_debug(f"成功创建数据集: {path}")
            
        except Exception as e:
            logger.error(f"创建数据集失败 {path}: {e}")
            # 尝试递归删除整个路径
            try:
                if path in h5file:
                    del h5file[path]
                
                shape = data.shape
                if len(shape) >= 2 and shape[0] > 10:
                    h5file.create_dataset(path, data=data, dtype=dtype)
                else:
                    h5file.create_dataset(path, data=data, dtype=dtype)
                log_info(f"通过递归删除成功创建数据集: {path}", verbose_only=True)
            except Exception as e2:
                logger.error(f"递归删除后仍无法创建数据集 {path}: {e2}")
                raise
    
    def _save_config_info(self, hdf5_file):
        """保存配置信息"""
        hdf5_file.attrs['alignment_window'] = self.alignment_config.alignment_window
        hdf5_file.attrs['target_fps'] = self.alignment_config.target_fps
        hdf5_file.attrs['sample_drop'] = self.alignment_config.sample_drop
        hdf5_file.attrs['relative_start'] = self.alignment_config.relative_start
        hdf5_file.attrs['delta_action'] = self.alignment_config.delta_action
        
        # 保存话题配置信息
        topic_info = {}
        for topic_name, config in self.topic_configs.items():
            topic_info[topic_name] = {
                'message_type': config.message_type,
                'hdf5_path': config.hdf5_path,
                'data_type': config.data_type,
                'description': config.description
            }
        
        hdf5_file.attrs['topic_configs'] = yaml.dump(topic_info)
    
    
    def _save_warning_stats_to_hdf5(self, hdf5_file, warning_stats: Dict, original_data: Dict = None, original_mcap_file: str = None):
        """将报警统计信息保存到HDF5文件中"""
        try:
            import json
            from datetime import datetime
            
            # 创建报警统计报告
            strategy_name = self.alignment_config.strategy
            if strategy_name == 'backfill_on_grid':
                description = 'Warning statistics for backfill_on_grid strategy'
            elif strategy_name == 'hybrid_alignment':
                description = 'Warning statistics for hybrid alignment strategy (image backfill + numerical interpolation)'
            else:
                description = f'Warning statistics for {strategy_name} strategy'
                
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategy': strategy_name,
                'hdf5_file': str(Path(hdf5_file.filename).name),
                'original_mcap_file': str(Path(original_mcap_file).name) if original_mcap_file else 'unknown',
                'description': description,
                'warnings': {}
            }
            
            # 统计各话题的报警信息
            total_warnings = 0
            for topic_name, warnings in warning_stats.items():
                if warnings:
                    topic_warnings = {
                        'count': len(warnings),
                        'details': warnings
                    }
                    report['warnings'][topic_name] = topic_warnings
                    total_warnings += len(warnings)
            
            report['total_warnings'] = total_warnings
            
            # 添加话题结束时间分析
            if original_data:
                topic_end_analysis = self._calculate_topic_end_analysis(original_data)
                if topic_end_analysis:
                    report['topic_end_analysis'] = topic_end_analysis
            
            # 将报告转换为JSON字符串并保存到HDF5文件中
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            
            # 创建字符串数据集
            hdf5_file.create_dataset('/metadata/warning_stats', data=report_json.encode('utf-8'))
            
            log_info(f"Warning statistics saved to HDF5 file: /metadata/warning_stats (total {total_warnings} warnings)", verbose_only=True)
            
        except Exception as e:
            log_warning(f"Failed to save warning statistics to HDF5: {e}", verbose_only=True)


def _map_ros_type_to_converter_type(ros_type: str) -> str:
    """Map ROS message type to converter internal type"""
    if 'Image' in ros_type:
        if 'Compressed' in ros_type:
            return 'compressed_image'
        return 'image'
    elif 'JointState' in ros_type:
        return 'joint_state'
    elif 'TwistStamped' in ros_type:
        return 'twist_stamped'
    elif 'Twist' in ros_type:
        return 'twist'
    elif 'PoseStamped' in ros_type:
        return 'pose_stamped'
    elif 'Float32MultiArray' in ros_type:
        return 'float32_multiarray'
    elif 'Float64MultiArray' in ros_type:
        return 'float64_multiarray'
    elif 'Float32' in ros_type:
        return 'float32'
    elif 'Float64' in ros_type:
        return 'float64'
    elif 'Sixforce' in ros_type:
        return 'sixforce'
    elif 'Int32' in ros_type:
        return 'int32'
    return 'custom'

def _infer_semantic_root(topic_name: str, converter_type: str) -> str:
    """
    根据 topic/类型推断是“观测 observations”还是“动作/指令 actions”。
    规则尽量保守：优先保证已显式 observations/actions 的路径不被覆盖。
    """
    name = (topic_name or "").strip().lower()

    # 图像通常属于观测
    if converter_type in ["compressed_image", "image"]:
        return "observations"

    # 关节反馈/状态：观测
    if "joint_state" in name or "joint_states" in name:
        return "observations"

    # 常见指令/目标：动作
    action_markers = [
        "joint_cmd",
        "gripper_cmd",
        "gripper_target",
        "target_",
        "/target_",
        "target",
        "cmd",
        "command",
        "teleop",
        "controller",
    ]
    if any(m in name for m in action_markers):
        return "actions"

    # 兜底：未知标量/速度/位姿倾向按观测理解
    return "observations"

def _extract_camera_key(topic_name: str) -> str:
    """
    为图像话题提取一个稳定的 camera_key，用于生成：
      observations/images/{camera_key}
    """
    name = (topic_name or "").strip().lstrip("/")
    lower = name.lower()
    parts = [p for p in name.split("/") if p]

    # 优先使用更“语义化”的前缀
    if "realsense_left_hand" in lower:
        return "realsense_left_hand"
    if "realsense_right_hand" in lower:
        return "realsense_right_hand"
    if "usb_cam_fisheye" in lower:
        return "usb_cam_fisheye"
    if "usb_cam_left" in lower:
        return "usb_cam_left"
    if "usb_cam_right" in lower:
        return "usb_cam_right"
    if "rgbd" in lower:
        return "rgbd"

    # 匹配 camera_01 / camera01 等
    m = re.search(r"(camera_?\d+)", lower)
    if m:
        return m.group(1).replace("-", "_")

    # 否则取第一个段落，保证可读且稳定
    return parts[0] if parts else "camera"

def _rewrite_hdf5_root_path(topic_name: str, converter_type: str, current_hdf5_path: str) -> str:
    """
    若当前 hdf5_path 未显式包含 observations/ 或 actions/ 顶层，则重写为：
      - 图像：observations/images/{camera_key}
      - 其他：{observations|actions}/{topic_name.lstrip('/') }
    """
    hp = (current_hdf5_path or "").strip()
    hp_norm = hp.lstrip("/")
    if hp_norm == "observations" or hp_norm.startswith("observations/") or hp_norm == "actions" or hp_norm.startswith("actions/"):
        return hp

    semantic_root = _infer_semantic_root(topic_name, converter_type)

    if converter_type in ["compressed_image", "image"]:
        camera_key = _extract_camera_key(topic_name)
        return f"{semantic_root}/images/{camera_key}"

    return f"{semantic_root}/{(topic_name or '').strip().lstrip('/')}"

def _normalize_topic_configs_roots(topic_configs: List["TopicConfig"]) -> None:
    # TopicConfig 在文件中定义为 dataclass，此处用字符串类型避免前向引用问题。
    for cfg in topic_configs:
        if not getattr(cfg, "hdf5_path", None):
            continue
        cfg.hdf5_path = _rewrite_hdf5_root_path(
            topic_name=getattr(cfg, "topic_name", ""),
            converter_type=getattr(cfg, "message_type", ""),
            current_hdf5_path=getattr(cfg, "hdf5_path", ""),
        )

def _scan_topics_and_create_config(mcap_path: str) -> List[Dict[str, Any]]:
    """Scan MCAP file for topics and create default configuration"""
    try:
        reader = make_reader(open(mcap_path, "rb"), decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        
        channels = {}
        schemas = {}
        
        if summary:
            channels = summary.channels
            schemas = summary.schemas
        else:
            logger.warning("No summary found in MCAP, scanning messages to find topics...")
            for schema, channel, message in reader.iter_messages():
                if channel.id not in channels:
                    channels[channel.id] = channel
                if schema and schema.id not in schemas:
                    schemas[schema.id] = schema
        
        topics_config = []
        # Priority mapping for common topics
        # Lower number means higher priority for main loop/visualization
        
        for channel_id, channel in channels.items():
            topic_name = channel.topic
            msg_type = channel.message_encoding
            
            if channel.schema_id in schemas:
                schema = schemas[channel.schema_id]
                ros_type = schema.name
            else:
                ros_type = "unknown"
            
            # Skip TF and other system topics
            if topic_name in ['/tf', '/tf_static', '/rosout', '/parameter_events']:
                continue
                
            converter_type = _map_ros_type_to_converter_type(ros_type)
            
            # Generate HDF5 path based on topic name
            # Here we rewrite to observations/actions at top-level by heuristic,
            # so annotation/preview can rely on a more consistent layout.
            converter_type = _map_ros_type_to_converter_type(ros_type)
            hdf5_path = _rewrite_hdf5_root_path(
                topic_name=topic_name,
                converter_type=converter_type,
                current_hdf5_path=topic_name,
            )
            
            config = {
                "topic_name": topic_name,
                "message_type": converter_type,
                "hdf5_path": hdf5_path,
                "data_type": "float32",  # Default
                "description": f"Auto-discovered from {ros_type}"
            }
            
            # Customize based on type
            if converter_type == 'compressed_image' or converter_type == 'image':
                config['data_type'] = 'uint8'
            elif converter_type in ['float64_multiarray', 'float64', 'sixforce']:
                config['data_type'] = 'float64'
            
            topics_config.append(config)
            
        return topics_config
    except Exception as e:
        logger.error(f"Failed to scan topics: {e}")
        return []

from app.services.inspect_mcap_freq import get_frequency_stats

def analyze_mcap_frequency(mcap_path: str) -> List[Dict[str, Any]]:
    """
    Analyze MCAP file to get frequency statistics for each topic.
    Returns a list of dictionaries containing topic stats.
    """
    try:
        results = get_frequency_stats(mcap_path)
        
        # Ensure values are rounded as expected by the frontend/API
        for stats in results:
             if 'frequency' in stats and isinstance(stats['frequency'], float):
                 stats['frequency'] = round(stats['frequency'], 2)
             if 'period_ms' in stats and isinstance(stats['period_ms'], float):
                 stats['period_ms'] = round(stats['period_ms'], 2)
             if 'min_delta_ms' in stats and isinstance(stats['min_delta_ms'], float):
                 stats['min_delta_ms'] = round(stats['min_delta_ms'], 2)
             if 'max_delta_ms' in stats and isinstance(stats['max_delta_ms'], float):
                 stats['max_delta_ms'] = round(stats['max_delta_ms'], 2)
                 
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing MCAP frequency: {e}")
        return []

def convert_mcap_to_hdf5(
    mcap_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[bool, str]:
    """
    主转换函数，将MCAP文件转换为HDF5格式。
    progress_callback(stage_name, percent) 可选，用于上报阶段进度（Parse/Align/Write/Validate）。
    返回 (是否成功, 失败原因)；成功时第二项为空字符串。
    """
    global verbose_mode

    # 处理频率设置 (frontend integration)
    if 'frequency' in config:
        if 'alignment' not in config:
            config['alignment'] = {}
        try:
            freq = float(config['frequency'])
            if freq > 0:
                config['alignment']['grid_fps'] = freq
                logger.info(f"Setting alignment grid_fps to {freq} from config")
        except (ValueError, TypeError):
            logger.warning(f"Invalid frequency in config: {config['frequency']}")

    # 检查topics参数
    topics = config.get('topics')
    
    # 情况1: 没有提供话题 -> 自动扫描所有
    # 情况2: 提供了话题名称列表 (List[str]) -> 扫描所有并过滤
    
    should_scan = False
    if not topics:
        should_scan = True
        logger.info("Config missing 'topics', performing auto-discovery...")
    elif isinstance(topics, list) and len(topics) > 0 and isinstance(topics[0], str):
        should_scan = True
        logger.info(f"Received {len(topics)} topic names, performing discovery to resolve configs...")
        
    if should_scan:
        discovered_topics = _scan_topics_and_create_config(mcap_path)
        
        if not topics:
            # 情况1: 使用所有发现的话题
            if discovered_topics:
                config['topics'] = discovered_topics
                logger.info(f"Auto-discovered {len(discovered_topics)} topics")
            else:
                logger.warning("Auto-discovery found no topics")
        elif isinstance(topics, list) and len(topics) > 0 and isinstance(topics[0], str):
            # 情况2: 过滤
            selected_names = set(topics)
            filtered_topics = [t for t in discovered_topics if t['topic_name'] in selected_names]
            
            if filtered_topics:
                config['topics'] = filtered_topics
                logger.info(f"Resolved {len(filtered_topics)} topic configs from names")
            else:
                logger.warning("No topics matched user selection, falling back to all discovered topics")
                config['topics'] = discovered_topics

    # 从字典加载配置
    try:
        topic_configs = [TopicConfig(**cfg) for cfg in config.get('topics', [])]
        # 统一 observation/action 顶层结构（除非调用方已显式指定）
        _normalize_topic_configs_roots(topic_configs)
        alignment_config = AlignmentConfig(**config.get('alignment', {}))
        data_merging_config = config.get('data_merging', {})
        
        # 创建性能监控器
        monitor = PerformanceMonitor()
        monitor.start()
        
        if progress_callback:
            progress_callback("Parse", 10)
        # 1. 处理MCAP文件
        reader = FlexibleMcapReader(topic_configs)
        data = reader.process_mcap(mcap_path, monitor)
        if progress_callback:
            progress_callback("Parse", 25)
        if not data:
            log_warning(f"没有从MCAP文件中读取到任何数据: {mcap_path}")
            return False, "没有从 MCAP 读取到任何数据（文件无法打开、无匹配话题或消息解析失败）"

        topics_with_msgs = {t for t, msgs in data.items() if msgs}
        if not topics_with_msgs:
            log_warning(f"MCAP 已解析但各话题消息数为 0: {mcap_path}")
            return False, "各配置话题均无有效消息，无法对齐与导出（请检查话题名与消息类型是否匹配）"
        
        if progress_callback:
            progress_callback("Align", 45)
        # 2. 时间戳对齐（仅保留 backfill_on_grid）
        monitor.mark("timestamp_alignment")
        aligned_data, alignment_details, dropped_counts, window_size, warning_stats = reader.align_data_backfill_on_grid(
            data, alignment_config)
        if progress_callback:
            progress_callback("Align", 55)
        # 3. 转换到HDF5（aligned_data 可能为空，写入逻辑会为未匹配话题创建空数据集）
        monitor.mark("hdf5_conversion")
        converter = FlexibleHdf5Converter(topic_configs, alignment_config, data_merging_config)
        success = converter.convert_to_hdf5(aligned_data, output_path, alignment_details, dropped_counts, window_size, warning_stats, data, mcap_path)
        if progress_callback:
            progress_callback("Write", 85)
        if not success:
            logger.error(f"HDF5 转换失败: {output_path}")
            return False, "HDF5 写入失败（详见服务端日志中的 HDF5转换失败 记录）"
        
        # 4. 打印性能指标
        monitor.mark("finish")
        monitor.print_metrics(mcap_path, output_path)
        
        return True, ""
        
    except Exception as e:
        logger.error(f"转换文件失败 {mcap_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False, f"{type(e).__name__}: {e}"


