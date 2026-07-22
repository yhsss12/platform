#!/usr/bin/env python3

import sys
import logging
import numpy as np
import h5py
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import cv2
import tqdm
import os
import glob
from dataclasses import dataclass
import time
import tracemalloc
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
    # 新增：对齐策略
    # window_strict: 现有窗口内严格对齐
    # backfill_on_grid: 等间隔时间网格上回填最近历史数据
    strategy: str = "window_strict"
    grid_fps: float = 15.0
    expand_start: bool = False
    # 插值配置
    interpolation_points: int = 2  # 插值使用的点数（前后各几个点，默认2表示前后各1个点）
    interpolation_method: str = "linear"  # 插值方法：linear, cubic, spline
    # 目标数据时长（秒），None表示不限制
    target_duration: Optional[float] = None

class FlexibleMcapProcessor:
    """灵活的消息处理器"""
    
    def __init__(self):
        self.processors = {
            'joint_state': self._process_joint_state,
            'compressed_image': self._process_compressed_image,
            'image': self._process_image,
            'float64_multiarray': self._process_float64_multiarray,
            'float32_multiarray': self._process_float32_multiarray,
            'sixforce': self._process_sixforce,
            'twist': self._process_twist,
            'twist_stamped': self._process_twist_stamped,
            'pose_stamped': self._process_pose_stamped,
            'int32': self._process_int32,
            'float32': self._process_float32,
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
                return {
                    'data': depth_array.astype(depth_dtype),
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
        """处理 rm_ros_interfaces/msg/Sixforce

        期望字段：
          - force_fx, force_fy, force_fz
          - force_mx, force_my, force_mz
        """
        try:
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

            # 维度固定为 6（缺失字段补 0），方便 LeRobot 侧直接当作 6D 向量使用
            values = [0.0 if v is None else float(v) for v in values]
            return {
                'data': np.asarray(values, dtype=np.float64),  # (6,)
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

    def _process_float32(self, msg, config: TopicConfig) -> Dict[str, Any]:
        """处理 std_msgs/Float32 消息（如夹爪开度）"""
        try:
            value = float(msg.data)
            data_array = np.array([value], dtype=getattr(np, config.data_type, np.float32))
            return {
                'data': data_array,
                'value': value
            }
        except Exception as e:
            log_debug(f"Float32处理失败: {e}")
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
                # 动态导入自定义处理器
                import importlib.util
                spec = importlib.util.spec_from_file_location("custom_processor", config.custom_processor)
                custom_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_module)
                
                if hasattr(custom_module, 'process_message'):
                    return custom_module.process_message(msg, config)
            
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
                decoder_factory = DecoderFactory()
                try:
                    # 正常路径：带 footer 的标准 MCAP
                    reader = make_reader(f, decoder_factories=[decoder_factory])
                    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                        total_messages += 1
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
                    # 兼容：部分 MCAP 文件缺少 footer（SeekingReader 会报 expected footer）
                    # 这种情况下用 StreamReader 顺序读取 records 并手动解码 Message。
                    if "expected footer at end of MCAP file" not in str(e):
                        raise
                    log_warning(f"检测到MCAP缺少footer，启用StreamReader兼容模式: {e}")

                    from mcap.stream_reader import StreamReader, Schema as SRSchema, Channel as SRChannel, Message as SRMessage
                    from mcap.exceptions import EndOfFile

                    # 重新从文件头开始读
                    f.seek(0)
                    sr = StreamReader(f)
                    schemas_by_id = {}
                    channels_by_id = {}

                    try:
                        for rec in sr.records:
                            if isinstance(rec, SRSchema):
                                schemas_by_id[rec.id] = rec
                                continue
                            if isinstance(rec, SRChannel):
                                channels_by_id[rec.id] = rec
                                continue
                            if not isinstance(rec, SRMessage):
                                continue

                            ch = channels_by_id.get(rec.channel_id)
                            if ch is None:
                                continue
                            topic = ch.topic
                            total_messages += 1

                            # 首先检查精确匹配
                            if topic in self.topic_configs:
                                config = self.topic_configs[topic]
                            else:
                                config = self._match_topic_patterns(topic)
                                if config:
                                    if topic not in self.dynamic_configs:
                                        self.dynamic_configs[topic] = config
                                        log_info(f"动态发现话题: {topic} (匹配模式: {config.topic_patterns})")
                                    original_topic = config.topic_name
                                    if original_topic not in self.data:
                                        self.data[original_topic] = []
                                    topic = original_topic
                                else:
                                    if topic not in skipped_topics:
                                        skipped_topics.add(topic)
                                        log_info(f"跳过未配置的话题: {topic}", verbose_only=True)
                                    continue

                            try:
                                schema = schemas_by_id.get(ch.schema_id)
                                decode = decoder_factory.decoder_for(ch.message_encoding, schema)
                                if decode is None:
                                    continue
                                ros_msg = decode(rec.data)

                                detected_type = self._auto_detect_message_type(ros_msg, config)
                                if detected_type != config.message_type:
                                    log_info(f"自动检测消息类型: {topic} {config.message_type} -> {detected_type}", verbose_only=True)
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

                                processor_key = config.custom_processor or config.message_type
                                processor = self.processor.processors.get(
                                    processor_key,
                                    self.processor.processors['custom']
                                )
                                msg_data = processor(ros_msg, config)
                                if msg_data is not None:
                                    msg_data["timestamp"] = rec.log_time / 1e9
                                    self.data[topic].append(msg_data)
                                    processed_messages += 1
                            except Exception as e2:
                                error_count += 1
                                if error_count <= 10:
                                    log_warning(f"处理消息时出错 {topic}: {e2}", verbose_only=True)
                                else:
                                    log_debug(f"处理消息时出错 {topic}: {e2}")
                                continue
                    except EndOfFile:
                        pass
                    
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
    
    def _save_failure_log(self, mcap_file: str, reason: str):
        """保存转换失败日志"""
        try:
            from datetime import datetime
            import json
            from pathlib import Path
            
            # 确定输出目录（与MCAP文件同目录）
            mcap_path = Path(mcap_file)
            report_dir = mcap_path.parent
            
            # 生成失败日志文件名
            base_name = mcap_path.stem
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f"{base_name}_failure_{ts}.json"
            
            payload = {
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'mcap_file': str(mcap_path.name),
                'reason': reason,
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            log_info(f"转换失败日志已保存: {report_file}", verbose_only=True)
        except Exception as e:
            log_warning(f"保存转换失败日志时出错: {e}", verbose_only=True)
    
    def align_data_with_window(self, data: Dict[str, List[Dict[str, Any]]], 
                             alignment_config: AlignmentConfig) -> Dict[str, List[Dict[str, Any]]]:
        """使用窗口对齐数据"""
        log_info("开始时间戳对齐...")
        
        # 选择主时间线
        if alignment_config.main_timeline_topic:
            main_topic = alignment_config.main_timeline_topic
            if main_topic not in data or len(data[main_topic]) == 0:
                logger.error(f"指定的主时间线话题 {main_topic} 不存在或为空")
                return {}
        else:
            # 自动选择数据量最小的话题作为主时间线
            valid_topics = {k: v for k, v in data.items() if len(v) > 0}
            if not valid_topics:
                logger.error("没有找到有效的话题数据")
                return {}
            
            main_topic = min(valid_topics.keys(), key=lambda k: len(valid_topics[k]))
            log_info(f"自动选择主时间线: {main_topic} (数据量: {len(valid_topics[main_topic])})")
        
        # 获取主时间线的时间戳
        main_timestamps = [msg['timestamp'] for msg in data[main_topic]]
        log_info(f"主时间线原始时间戳数量: {len(main_timestamps)}", verbose_only=True)
        
        # 应用采样和丢弃
        if alignment_config.sample_drop > 0:
            main_timestamps = main_timestamps[alignment_config.sample_drop:-alignment_config.sample_drop]
            log_info(f"应用丢弃首尾 {alignment_config.sample_drop} 帧后: {len(main_timestamps)}", verbose_only=True)
        
        # 应用目标帧率采样
        if alignment_config.target_fps > 0:
            # 计算采样间隔
            if len(main_timestamps) > 1:
                original_fps = 1.0 / (main_timestamps[1] - main_timestamps[0])
                log_info(f"原始帧率: {original_fps:.2f} Hz, 目标帧率: {alignment_config.target_fps} Hz", verbose_only=True)
                if original_fps > alignment_config.target_fps:
                    jump = int(original_fps / alignment_config.target_fps)
                    main_timestamps = main_timestamps[::jump]
                    log_info(f"应用帧率采样，跳跃间隔: {jump}, 采样后: {len(main_timestamps)}", verbose_only=True)
        
        # 暂时忽略结束时间过滤，保留更多数据
        # if data:
        #     min_end_time = min([data[k][-1]['timestamp'] for k in data.keys() if len(data[k]) > 0])
        #     logger.info(f"最短话题结束时间: {min_end_time}")
        #     original_count = len(main_timestamps)
        #     main_timestamps = [t for t in main_timestamps if t < min_end_time]
        #     logger.info(f"过滤结束时间后: {original_count} -> {len(main_timestamps)}")
        
        log_info(f"主时间线最终时间戳数量: {len(main_timestamps)}")
        
        # 窗口对齐 - 严格对齐
        aligned_data = defaultdict(list)
        window_size = alignment_config.alignment_window
        dropped_counts = defaultdict(int)  # 记录每个话题舍弃的数据量
        alignment_details = defaultdict(list)  # 记录详细的对齐信息
        
        # 首先过滤出有数据的话题，只对这些话题进行严格对齐
        topics_with_data = {topic: data for topic, data in data.items() if len(data) > 0}
        log_info(f"有数据的话题数量: {len(topics_with_data)}/{len(data)}", verbose_only=True)
        log_info(f"有数据的话题: {list(topics_with_data.keys())}", verbose_only=True)
        
        for target_timestamp in main_timestamps:
            aligned_row = {}
            all_topics_have_data = True
            
            # 只检查有数据的话题
            for topic_name, topic_data in topics_with_data.items():
                # 在窗口内查找最接近的消息
                topic_timestamps = [msg['timestamp'] for msg in topic_data]
                time_array = np.array(topic_timestamps)
                
                # 找到窗口内的所有时间戳
                window_mask = np.abs(time_array - target_timestamp) <= window_size
                window_indices = np.where(window_mask)[0]
                
                if len(window_indices) > 0:
                    # 在窗口内选择最接近的
                    window_times = time_array[window_indices]
                    closest_idx = window_indices[np.argmin(np.abs(window_times - target_timestamp))]
                    selected_timestamp = topic_timestamps[closest_idx]
                    time_diff = selected_timestamp - target_timestamp
                    
                    # 记录对齐详情
                    alignment_details[topic_name].append({
                        'target_time': target_timestamp,
                        'selected_time': selected_timestamp,
                        'time_diff': time_diff,
                        'abs_time_diff': abs(time_diff)
                    })
                    
                    aligned_row[topic_name] = topic_data[closest_idx]
                else:
                    # 窗口内没有数据，标记为不完整
                    all_topics_have_data = False
                    dropped_counts[topic_name] += 1
            
            # 只有当所有有数据的话题都能对齐时，才添加这一行
            if all_topics_have_data:
                for topic_name, data in aligned_row.items():
                    aligned_data[topic_name].append(data)
            else:
                # 舍弃整个时间步
                dropped_counts['all_topics'] += 1
        
        # 输出详细的对齐信息
        self._print_alignment_details(alignment_details, dropped_counts, window_size)
        
        log_info(f"时间戳对齐完成，对齐后各话题数据量:")
        for topic, data_list in aligned_data.items():
            original_count = len(data[topic]) if topic in data else 0
            dropped_count = dropped_counts[topic]
            log_info(f"  {topic}: {original_count} -> {len(data_list)} 条消息 (舍弃 {dropped_count} 条)")
        
        # 显示严格对齐的统计信息
        total_dropped_steps = dropped_counts.get('all_topics', 0)
        log_info(f"严格对齐统计:")
        log_info(f"  总时间步数: {len(main_timestamps)}")
        log_info(f"  成功对齐步数: {len(aligned_data[list(aligned_data.keys())[0]]) if aligned_data else 0}")
        log_info(f"  舍弃时间步数: {total_dropped_steps}")
        log_info(f"  对齐成功率: {len(aligned_data[list(aligned_data.keys())[0]]) / len(main_timestamps) * 100:.1f}%" if aligned_data and main_timestamps else "0%")
        
        return dict(aligned_data), dict(alignment_details), dict(dropped_counts)

    def align_data_backfill_on_grid(self, data: Dict[str, List[Dict[str, Any]]], 
                                    alignment_config: AlignmentConfig) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict, Dict]:
        """等间隔网格回填对齐（backfill_on_grid）。"""
        log_info("开始 backfill_on_grid 对齐...")
        topics_with_data = {t: msgs for t, msgs in data.items() if len(msgs) > 0}
        if not topics_with_data:
            logger.error("没有任何可用数据用于对齐")
            return {}, {}, {}, 0.0, {}

        topic_starts = {t: min(m['timestamp'] for m in msgs) for t, msgs in topics_with_data.items()}
        topic_ends = {t: max(m['timestamp'] for m in msgs) for t, msgs in topics_with_data.items()}
        grid_start = max(topic_starts.values())
        grid_end = max(topic_ends.values())
        if grid_end <= grid_start:
            log_warning("网格时间范围无效，无法进行对齐", verbose_only=True)
            return {}, {}, {}, 0.0, {}

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
        window_size = 1.0 / (alignment_config.grid_fps if alignment_config.grid_fps > 0 else 15.0)
        return (
            dict(aligned_data),
            dict(alignment_details),
            dict(dropped_counts),
            window_size,
            dict(warning_stats),
        )
    
    def align_data_hybrid(self, data: Dict[str, List[Dict[str, Any]]], 
                         alignment_config: AlignmentConfig) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict, Dict]:
        """混合对齐策略：图像数据使用等间隔回填，其他话题使用插值对齐。"""
        log_info("开始混合对齐策略（图像回填+数值插值）...")
        topics_with_data = {t: msgs for t, msgs in data.items() if len(msgs) > 0}
        if not topics_with_data:
            logger.error("没有任何可用数据用于对齐")
            return {}, {}, {}

        # 分离图像话题和其他话题
        image_topics = {}
        other_topics = {}
        
        for topic, msgs in topics_with_data.items():
            # 检查是否为图像话题（通过消息类型或话题名称判断）
            is_image_topic = False
            if topic in self.topic_configs:
                config = self.topic_configs[topic]
                if config.message_type in ['compressed_image', 'image']:
                    is_image_topic = True
                elif 'camera' in topic.lower() or 'image' in topic.lower():
                    is_image_topic = True
            
            if is_image_topic:
                image_topics[topic] = msgs
            else:
                other_topics[topic] = msgs
        
        log_info(f"图像话题数量: {len(image_topics)}")
        log_info(f"其他话题数量: {len(other_topics)}")
        log_info(f"图像话题: {list(image_topics.keys())}", verbose_only=True)
        log_info(f"其他话题: {list(other_topics.keys())}", verbose_only=True)

        # 计算网格时间范围
        all_topics = {**image_topics, **other_topics}
        topic_starts = {t: min(m['timestamp'] for m in msgs) for t, msgs in all_topics.items()}
        topic_ends = {t: max(m['timestamp'] for m in msgs) for t, msgs in all_topics.items()}
        grid_start = max(topic_starts.values())
        grid_end = max(topic_ends.values())
        if grid_end <= grid_start:
            log_warning("网格时间范围无效，无法进行对齐", verbose_only=True)
            return {}, {}, {}

        fps = alignment_config.grid_fps if alignment_config.grid_fps and alignment_config.grid_fps > 0 else 15.0
        period = 1.0 / fps
        grid_times: List[float] = []
        t = grid_start
        while t <= grid_end + 1e-9:
            grid_times.append(t)
            t += period
        log_info(f"网格步数: {len(grid_times)}, 频率: {fps}Hz, 范围: [{grid_start:.6f}, {grid_end:.6f}]")

        # 预计算每个话题的时间戳数组
        topic_ts: Dict[str, np.ndarray] = {}
        topic_data: Dict[str, np.ndarray] = {}  # 存储原始数据用于插值
        
        for topic, msgs in all_topics.items():
            topic_ts[topic] = np.array([m['timestamp'] for m in msgs], dtype=np.float64)
            # 提取数据部分用于插值
            if topic in image_topics:
                # 图像数据保持原始结构
                topic_data[topic] = msgs
            else:
                # 其他话题提取数值数据
                topic_data[topic] = np.array([m['data'] for m in msgs])

        aligned_data = defaultdict(list)
        alignment_details = defaultdict(list)
        dropped_counts = defaultdict(int)
        warning_stats = defaultdict(list)
        
        for target_t in grid_times:
            row: Dict[str, Dict[str, Any]] = {}
            all_found = True
            
            # 处理图像话题：使用回填策略
            for topic, msgs in image_topics.items():
                ts = topic_ts[topic]
                start_t = target_t - period
                right = np.searchsorted(ts, target_t, side='right')
                left = np.searchsorted(ts, start_t, side='left')
                chosen_idx = None
                over_period = False
                
                if right - left > 0:
                    chosen_idx = right - 1
                else:
                    if right > 0:
                        chosen_idx = right - 1
                        over_period = True
                    else:
                        if alignment_config.expand_start and len(ts) > 0:
                            chosen_idx = 0
                            over_period = True
                        else:
                            all_found = False
                            dropped_counts[topic] += 1
                            continue

                if chosen_idx is not None:
                    row[topic] = msgs[chosen_idx]
                    sel_t = float(topic_ts[topic][chosen_idx])
                    detail = {
                        'target_time': target_t,
                        'selected_time': sel_t,
                        'time_diff': sel_t - target_t,
                        'abs_time_diff': abs(sel_t - target_t),
                        'method': 'backfill'
                    }
                    alignment_details[topic].append(detail)
            
            # 处理其他话题：使用插值策略
            for topic, msgs in other_topics.items():
                ts = topic_ts[topic]
                data_array = topic_data[topic]
                
                # 使用增强的插值方法
                interpolated_data, detail = self._interpolate_with_multiple_points(
                    ts, data_array, target_t, 
                    alignment_config.interpolation_points, 
                    alignment_config.interpolation_method
                )
                
                if interpolated_data is not None:
                    # 找到对应的原始消息作为模板
                    right_idx = np.searchsorted(ts, target_t, side='right')
                    left_idx = right_idx - 1
                    template_idx = left_idx if left_idx >= 0 else right_idx
                    template_msg = msgs[template_idx].copy()
                    template_msg['data'] = interpolated_data
                    template_msg['timestamp'] = target_t
                    
                    row[topic] = template_msg
                    alignment_details[topic].append(detail)
                else:
                    # 没有数据
                    all_found = False
                    dropped_counts[topic] += 1
                    detail.update({
                        'selected_time': None,
                        'time_diff': None,
                        'abs_time_diff': None,
                        'warning': 'No data available for interpolation'
                    })
                    alignment_details[topic].append(detail)
                    continue

            if all_found and row:
                for topic, msg in row.items():
                    aligned_data[topic].append(msg)
            else:
                dropped_counts['all_topics'] += 1

        log_info("混合对齐策略完成")
        window_size = 1.0 / (alignment_config.grid_fps if alignment_config.grid_fps > 0 else 15.0)
        return (
            dict(aligned_data),
            dict(alignment_details),
            dict(dropped_counts),
            window_size,
            dict(warning_stats),
        )
    
    def _interpolate_with_multiple_points(self, timestamps: np.ndarray, data: np.ndarray, 
                                        target_time: float, points: int, method: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        使用多个点进行插值
        
        Args:
            timestamps: 时间戳数组
            data: 对应的数据数组
            target_time: 目标时间点
            points: 插值使用的总点数
            method: 插值方法
            
        Returns:
            (插值结果, 详细信息)
        """
        if len(timestamps) == 0 or len(data) == 0:
            return None, {'error': 'No data available'}
        
        # 找到目标时间点的位置
        right_idx = np.searchsorted(timestamps, target_time, side='right')
        left_idx = right_idx - 1
        
        # 计算需要的点数
        half_points = points // 2
        
        # 收集插值点
        interpolation_points = []
        interpolation_times = []
        
        # 收集左侧点
        for i in range(half_points):
            idx = left_idx - i
            if idx >= 0:
                interpolation_points.append(data[idx])
                interpolation_times.append(timestamps[idx])
        
        # 收集右侧点
        for i in range(half_points):
            idx = right_idx + i
            if idx < len(timestamps):
                interpolation_points.append(data[idx])
                interpolation_times.append(timestamps[idx])
        
        # 按时间排序
        if interpolation_times:
            sorted_indices = np.argsort(interpolation_times)
            interpolation_times = [interpolation_times[i] for i in sorted_indices]
            interpolation_points = [interpolation_points[i] for i in sorted_indices]
        
        # 检查是否有足够的点进行插值
        if len(interpolation_points) < 2:
            # 使用最近邻
            if left_idx >= 0:
                return data[left_idx], {'method': 'nearest_neighbor_left', 'points_used': 1}
            elif right_idx < len(timestamps):
                return data[right_idx], {'method': 'nearest_neighbor_right', 'points_used': 1}
            else:
                return None, {'error': 'No data available for interpolation'}
        
        # 转换为numpy数组
        interpolation_times = np.array(interpolation_times)
        interpolation_points = np.array(interpolation_points)
        
        # 执行插值
        try:
            if method == "linear":
                result = self._linear_interpolation(interpolation_times, interpolation_points, target_time)
            elif method == "cubic":
                result = self._cubic_interpolation(interpolation_times, interpolation_points, target_time)
            elif method == "spline":
                result = self._spline_interpolation(interpolation_times, interpolation_points, target_time)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            detail = {
                'method': method,
                'points_used': len(interpolation_points),
                'interpolation_times': interpolation_times.tolist(),
                'target_time': target_time
            }
            
            return result, detail
            
        except Exception as e:
            log_warning(f"插值失败，使用最近邻: {e}", verbose_only=True)
            # 回退到最近邻
            if left_idx >= 0:
                return data[left_idx], {'method': 'nearest_neighbor_fallback', 'error': str(e)}
            elif right_idx < len(timestamps):
                return data[right_idx], {'method': 'nearest_neighbor_fallback', 'error': str(e)}
            else:
                return None, {'error': f'Interpolation failed: {e}'}
    
    def _linear_interpolation(self, times: np.ndarray, data: np.ndarray, target_time: float) -> np.ndarray:
        """线性插值"""
        if len(times) == 2:
            # 简单的两点线性插值
            t1, t2 = times[0], times[1]
            d1, d2 = data[0], data[1]
            if abs(t2 - t1) < 1e-9:
                return d1
            alpha = (target_time - t1) / (t2 - t1)
            return (1 - alpha) * d1 + alpha * d2
        else:
            # 多点线性插值（分段线性）
            return np.interp(target_time, times, data)
    
    def _cubic_interpolation(self, times: np.ndarray, data: np.ndarray, target_time: float) -> np.ndarray:
        """三次插值"""
        if len(times) < 4:
            # 点数不足，回退到线性插值
            return self._linear_interpolation(times, data, target_time)
        
        # 使用scipy的三次插值
        from scipy import interpolate
        f = interpolate.interp1d(times, data, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        return f(target_time)
    
    def _spline_interpolation(self, times: np.ndarray, data: np.ndarray, target_time: float) -> np.ndarray:
        """样条插值"""
        if len(times) < 3:
            # 点数不足，回退到线性插值
            return self._linear_interpolation(times, data, target_time)
        
        # 使用scipy的样条插值
        from scipy import interpolate
        tck = interpolate.splrep(times, data, s=0)  # s=0表示不进行平滑
        return interpolate.splev(target_time, tck)
    
    def _print_alignment_details(self, alignment_details: Dict, dropped_counts: Dict, window_size: float):
        """打印详细的时间戳对齐信息"""
        if not verbose_mode:
            return  # 非详细模式下不打印详细信息
            
        logger.info("=" * 80)
        logger.info("时间戳对齐详细信息")
        logger.info("=" * 80)
        logger.info(f"对齐窗口大小: {window_size:.6f} 秒")
        logger.info("")
        
        for topic_name, details in alignment_details.items():
            if not details:
                continue
                
            logger.info(f"话题: {topic_name}")
            logger.info("-" * 60)
            
            # 计算统计信息
            time_diffs = [d['time_diff'] for d in details]
            abs_time_diffs = [d['abs_time_diff'] for d in details]
            
            min_diff = min(time_diffs)
            max_diff = max(time_diffs)
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            
            min_abs_diff = min(abs_time_diffs)
            max_abs_diff = max(abs_time_diffs)
            mean_abs_diff = np.mean(abs_time_diffs)
            
            logger.info(f"  对齐数据量: {len(details)} 条")
            logger.info(f"  舍弃数据量: {dropped_counts[topic_name]} 条")
            logger.info(f"  时间差值统计:")
            logger.info(f"    最小差值: {min_diff:.6f} 秒")
            logger.info(f"    最大差值: {max_diff:.6f} 秒")
            logger.info(f"    平均差值: {mean_diff:.6f} 秒")
            logger.info(f"    标准差:   {std_diff:.6f} 秒")
            logger.info(f"  绝对时间差值统计:")
            logger.info(f"    最小绝对差值: {min_abs_diff:.6f} 秒")
            logger.info(f"    最大绝对差值: {max_abs_diff:.6f} 秒")
            logger.info(f"    平均绝对差值: {mean_abs_diff:.6f} 秒")
            
            # 显示前10个对齐示例
            logger.info(f"  前10个对齐示例:")
            for i, detail in enumerate(details[:10]):
                logger.info(f"    [{i+1:2d}] 目标时间: {detail['target_time']:.6f}s, "
                          f"选择时间: {detail['selected_time']:.6f}s, "
                          f"差值: {detail['time_diff']:+.6f}s")
            
            if len(details) > 10:
                logger.info(f"    ... 还有 {len(details) - 10} 个对齐点")
            
            logger.info("")
        
        logger.info("=" * 80)
    
    def _save_alignment_details(self, alignment_details: Dict, dropped_counts: Dict, window_size: float, output_hdf5_path: str = None):
        """保存详细的对齐信息到文件，与HDF5文件保存在同一目录"""
        import json
        from datetime import datetime
        import os
        from pathlib import Path
        
        # 创建对齐详情报告（添加中文注释）
        report = {
            'timestamp': datetime.now().isoformat(),
            'alignment_window': window_size,
            '注释': '本文件记录了时间戳对齐的详细差值信息，单位为秒',
            'topics': {}
        }
        
        for topic_name, details in alignment_details.items():
            if not details:
                continue
            
            # 计算统计信息
            time_diffs = [d['time_diff'] for d in details]
            abs_time_diffs = [d['abs_time_diff'] for d in details]
            
            topic_report = {
                'aligned_count': len(details),
                'dropped_count': dropped_counts[topic_name],
                '说明': 'aligned_count 为成功对齐的数据点数量，dropped_count 为因超出窗口被舍弃的数量',
                'statistics': {
                    'min_diff': float(min(time_diffs)),
                    'max_diff': float(max(time_diffs)),
                    'mean_diff': float(np.mean(time_diffs)),
                    'std_diff': float(np.std(time_diffs)),
                    'min_abs_diff': float(min(abs_time_diffs)),
                    'max_abs_diff': float(max(abs_time_diffs)),
                    'mean_abs_diff': float(np.mean(abs_time_diffs))
                },
                'statistics_注释': 'diff 为(选择时间-目标时间)，abs为绝对值；越小表示对齐越好',
                'alignment_details': details,
                'alignment_details_注释': '每个条目包含 target_time(目标时间), selected_time(被选中的数据时间), time_diff(差值), abs_time_diff(差值绝对值)'
            }
            
            report['topics'][topic_name] = topic_report
        
        # 确定保存目录
        if output_hdf5_path:
            # 与HDF5文件保存在同一目录
            hdf5_path = Path(output_hdf5_path)
            report_dir = hdf5_path.parent
            # 使用HDF5文件名作为前缀
            base_name = hdf5_path.stem
            filename = f"{base_name}_alignment_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            # 回退到环境变量或当前目录
            report_dir = os.environ.get('ALIGNMENT_REPORT_DIR')
            if report_dir:
                try:
                    Path(report_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    report_dir = None
            else:
                report_dir = Path.cwd()
            filename = f"alignment_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = str(Path(report_dir) / filename)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            log_info(f"对齐详情已保存到文件: {output_file}", verbose_only=True)
        except Exception as e:
            log_warning(f"保存对齐详情文件失败: {e}", verbose_only=True)

        # 额外输出报警/错误信息日志文件（仅记录包含 warning 字段的条目）
        try:
            warning_lines: List[str] = []
            warning_lines.append("# backfill_on_grid 对齐报警/错误日志\n")
            warning_lines.append(f"hdf5_file: {Path(output_hdf5_path).name}\n")
            warning_lines.append(f"generated_at: {datetime.now().isoformat()}\n\n")
            
            warning_count = 0
            for topic_name, details in alignment_details.items():
                for d in details:
                    warn = d.get('warning')
                    if warn:
                        tgt = d.get('target_time')
                        sel = d.get('selected_time')
                        diff = d.get('time_diff')
                        # 同时输出中英文关键字段，便于人读与机器解析
                        warning_lines.append(
                            f"话题={topic_name} | 报警={warn} | 时间步={tgt:.6f}s | 选中时间={sel:.6f}s | 差值={diff:+.6f}s\n"
                        )
                        warning_count += 1
            
            if warning_count > 0:
                warn_filename = f"{base_name}_alignment_warnings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                warn_path = str(Path(report_dir) / warn_filename)
                with open(warn_path, 'w', encoding='utf-8') as wf:
                    wf.writelines(warning_lines)
                log_info(f"对齐报警日志已保存: {warn_path} (共{warning_count}条报警)", verbose_only=True)
            else:
                log_info("无报警信息需要保存（未出现跨周期或越过起点的回填）", verbose_only=True)
        except Exception as e:
            log_warning(f"保存对齐报警日志失败: {e}", verbose_only=True)

class FlexibleHdf5Converter:
    """灵活的HDF5转换器"""
    
    def __init__(self, topic_configs: List[TopicConfig], alignment_config: AlignmentConfig, 
                 data_merging_config: Dict[str, Any] = None):
        self.topic_configs = {config.topic_name: config for config in topic_configs}
        self.alignment_config = alignment_config
        self.data_merging_config = data_merging_config or {}

    @staticmethod
    def _normalize_image_base_path(hdf5_path: str) -> str:
        """
        兼容旧配置：
          - 旧版可能直接给到 /images/camera1/color/origin 或 /images/camera1/depth/origin
        新版写入会在 base 下创建：
          {base}/color/origin, {base}/color/compress, {base}/depth, {base}/pointcloud
        """
        hp = (hdf5_path or "").strip()
        hp = "/" + hp.lstrip("/")
        for suffix in ("/color/origin", "/color/compress"):
            if hp.endswith(suffix):
                return hp[: -len(suffix)]
        for suffix in ("/depth/origin", "/depth", "/pointcloud/origin", "/pointcloud"):
            if hp.endswith(suffix):
                return hp.rsplit("/", 1)[0] if suffix in ("/depth", "/pointcloud") else hp[: -len(suffix)]
        return hp
    
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
                    is_image_topic = (
                        config.message_type in ['compressed_image', 'image']
                        or (config.custom_processor in ['compressed_image', 'image'])
                    )
                    if is_image_topic:
                        hdf5_path = self._normalize_image_base_path(hdf5_path)
                    
                    if len(data_list) == 0:
                        # 即使没有数据，也要为所有类型的话题创建路径
                        if (
                            config.message_type in ['compressed_image', 'image']
                            or (config.custom_processor in ['compressed_image', 'image'])
                        ):
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
                    if (
                        config.message_type in ['compressed_image', 'image']
                        or (config.custom_processor in ['compressed_image', 'image'])
                    ):
                        # 图像数据 - 按照新结构保存
                        # 新结构：需要创建 color/origin, color/compress, depth, pointcloud 四个路径
                        # hdf5_path 应该是 /images/cam_head 这样的路径
                        origin_path = f"{hdf5_path}/color/origin"
                        compress_path = f"{hdf5_path}/color/compress"
                        depth_path = f"{hdf5_path}/depth"
                        pointcloud_path = f"{hdf5_path}/pointcloud"
                        
                        # 关键修复：避免 topic 顺序导致 color/depth 相互覆盖
                        has_color_data = any(msg.get('data') is not None for msg in data_list)
                        has_depth_data = any(msg.get('depth') is not None for msg in data_list)
                        has_pointcloud_data = any(msg.get('pointcloud') is not None for msg in data_list)

                        if has_color_data:
                            color_frames = [msg.get('data') for msg in data_list if msg.get('data') is not None]
                            img_array = np.stack(color_frames)
                            if config.message_type == 'compressed_image':
                                self._create_or_replace_dataset(f, compress_path, img_array, np.uint8)
                                log_info(f"保存压缩图像数据: {compress_path} {img_array.shape}", verbose_only=True)
                                self._ensure_image_group_no_overwrite(f, origin_path)
                            else:
                                self._create_or_replace_dataset(f, origin_path, img_array, np.uint8)
                                log_info(f"保存原始图像数据: {origin_path} {img_array.shape}", verbose_only=True)
                                self._ensure_image_group_no_overwrite(f, compress_path)
                        else:
                            # 深度 topic：不写 color；只在路径缺失时确保 group 存在
                            self._ensure_image_group_no_overwrite(f, origin_path)
                            self._ensure_image_group_no_overwrite(f, compress_path)

                        if has_depth_data:
                            depth_frames = [
                                np.asarray(msg.get('depth'), dtype=np.float32)
                                for msg in data_list
                                if msg.get('depth') is not None
                            ]
                            try:
                                depth_array = np.stack(depth_frames, axis=0)
                            except ValueError:
                                log_warning("深度图像形状不一致，仅保存首帧数据", verbose_only=True)
                                depth_array = depth_frames[0][None, ...]
                            self._create_or_replace_dataset(f, depth_path, depth_array, np.float32)
                            log_info(f"保存深度图像数据: {depth_path} {depth_array.shape}", verbose_only=True)
                        else:
                            self._ensure_empty_dataset_no_overwrite(f, depth_path, np.float32)
                        
                        if has_pointcloud_data:
                            pointcloud_frames = [
                                np.asarray(msg.get('pointcloud'), dtype=np.float32)
                                for msg in data_list
                                if msg.get('pointcloud') is not None
                            ]
                            try:
                                pointcloud_array = np.stack(pointcloud_frames, axis=0)
                            except ValueError:
                                log_warning("点云数据形状不一致，仅保存首帧数据", verbose_only=True)
                                pointcloud_array = pointcloud_frames[0][None, ...]
                            self._create_or_replace_dataset(f, pointcloud_path, pointcloud_array, np.float32)
                            log_info(f"保存点云数据: {pointcloud_path} {pointcloud_array.shape}", verbose_only=True)
                        else:
                            self._ensure_empty_dataset_no_overwrite(f, pointcloud_path, np.float32)
                    
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
                        # Sixforce：显式拆分 force / torque 子数据集，方便下游直接取用
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
                        if (
                            config.message_type in ['compressed_image', 'image']
                            or (config.custom_processor in ['compressed_image', 'image'])
                        ):
                            # 图像数据：创建 origin, compress, depth, pointcloud 四个路径
                            base = self._normalize_image_base_path(config.hdf5_path)
                            origin_path = f"{base}/color/origin"
                            compress_path = f"{base}/color/compress"
                            depth_path = f"{base}/depth"
                            pointcloud_path = f"{base}/pointcloud"
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
                
                # 保存报警信息到HDF5文件中
                if warning_stats is not None:
                    self._save_warning_stats_to_hdf5(f, warning_stats, original_data, original_mcap_file)
            
            # 保存对齐详情到与HDF5文件相同的目录（仅用于window_strict策略）
            if alignment_details and dropped_counts is not None and self.alignment_config.strategy == 'window_strict':
                self._save_alignment_details_with_hdf5(alignment_details, dropped_counts, window_size, output_path, warning_stats, original_data, original_mcap_file)
            
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
        start_time = None
        for topic_name, data_list in aligned_data.items():
            if len(data_list) > 0:
                topic_start = min(msg['timestamp'] for msg in data_list)
                if start_time is None or topic_start > start_time:
                    start_time = topic_start
        
        if start_time is None:
            log_warning("无法确定起始时间，跳过固定时长处理", verbose_only=True)
            return aligned_data
        
        target_end_time = start_time + target_duration
        log_info(
            f"应用目标时长限制: {target_duration:.2f}秒 "
            f"(起始时间: {start_time:.6f}, 目标结束时间: {target_end_time:.6f})"
        )
        
        result_data = {}
        for topic_name, data_list in aligned_data.items():
            if len(data_list) == 0:
                result_data[topic_name] = []
                continue
            
            # 按时间戳排序
            sorted_data = sorted(data_list, key=lambda x: x['timestamp'])
            
            # 找到在目标时长范围内的数据（从start_time到target_end_time）
            filtered_data = [msg for msg in sorted_data if start_time <= msg['timestamp'] <= target_end_time]
            
            # 如果话题的起始时间早于最晚起始时间，需要截取或向前填充
            topic_start = sorted_data[0]['timestamp'] if len(sorted_data) > 0 else None
            if topic_start is not None and topic_start < start_time:
                # 找到最接近start_time的数据点
                closest_idx = 0
                for i, msg in enumerate(sorted_data):
                    if msg['timestamp'] <= start_time:
                        closest_idx = i
                    else:
                        break
                
                # 如果最接近的数据点时间戳小于start_time，需要向前填充
                if sorted_data[closest_idx]['timestamp'] < start_time:
                    # 使用最接近的数据点作为模板，创建起始帧
                    start_frame = sorted_data[closest_idx].copy()
                    start_frame['timestamp'] = start_time
                    # 将起始帧插入到filtered_data的开头
                    if not filtered_data or filtered_data[0]['timestamp'] > start_time:
                        filtered_data.insert(0, start_frame)
            
            # 如果数据时长不足，使用最后一帧填充
            if len(filtered_data) > 0:
                last_timestamp = filtered_data[-1]['timestamp']
                if last_timestamp < target_end_time:
                    # 计算需要填充的帧数（基于平均帧间隔）
                    if len(filtered_data) > 1:
                        avg_interval = (filtered_data[-1]['timestamp'] - filtered_data[0]['timestamp']) / (len(filtered_data) - 1)
                    else:
                        # 如果只有一帧，使用目标帧率估算
                        avg_interval = 1.0 / (self.alignment_config.target_fps if self.alignment_config.target_fps > 0 else 30.0)
                    
                    remaining_duration = target_end_time - last_timestamp
                    num_padding = int(remaining_duration / avg_interval) if avg_interval > 0 else 0
                    
                    if num_padding > 0:
                        last_frame = filtered_data[-1].copy()
                        for i in range(num_padding):
                            padding_frame = last_frame.copy()
                            padding_frame['timestamp'] = last_timestamp + (i + 1) * avg_interval
                            if padding_frame['timestamp'] <= target_end_time:
                                filtered_data.append(padding_frame)
                            else:
                                break
                    
                    # 确保最后一帧的时间戳正好是目标结束时间
                    if len(filtered_data) > 0 and filtered_data[-1]['timestamp'] < target_end_time:
                        final_frame = filtered_data[-1].copy()
                        final_frame['timestamp'] = target_end_time
                        filtered_data[-1] = final_frame
                
                # 如果数据时长超出，截取到目标时长
                if filtered_data[-1]['timestamp'] > target_end_time:
                    filtered_data = [msg for msg in filtered_data if msg['timestamp'] <= target_end_time]
                    # 确保有最后一帧在目标结束时间
                    if len(filtered_data) > 0:
                        final_frame = filtered_data[-1].copy()
                        final_frame['timestamp'] = target_end_time
                        filtered_data[-1] = final_frame
            else:
                # 如果所有数据都在目标时长范围之外，使用最接近的数据点填充
                if len(sorted_data) > 0:
                    # 找到最接近start_time的数据点
                    closest_frame = sorted_data[0]
                    min_diff = abs(closest_frame['timestamp'] - start_time)
                    for msg in sorted_data:
                        diff = abs(msg['timestamp'] - start_time)
                        if diff < min_diff:
                            min_diff = diff
                            closest_frame = msg
                    
                    # 创建起始帧和结束帧
                    start_frame = closest_frame.copy()
                    start_frame['timestamp'] = start_time
                    end_frame = closest_frame.copy()
                    end_frame['timestamp'] = target_end_time
                    filtered_data = [start_frame, end_frame]
            
            result_data[topic_name] = filtered_data
            original_count = len(data_list)
            final_count = len(filtered_data)
            actual_duration = filtered_data[-1]['timestamp'] - filtered_data[0]['timestamp'] if len(filtered_data) > 1 else 0.0
            log_info(f"  话题 {topic_name}: {original_count} -> {final_count} 帧, 实际时长: {actual_duration:.2f}秒", verbose_only=True)
        
        return result_data
    
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
        避免“深度/点云 topic 因缺少 color 字段而覆盖已有彩色数据”的问题。
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
            h5file.create_dataset(path, data=data, dtype=dtype)
            log_debug(f"成功创建数据集: {path}")
            
        except Exception as e:
            logger.error(f"创建数据集失败 {path}: {e}")
            # 尝试递归删除整个路径
            try:
                if path in h5file:
                    del h5file[path]
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
    
    def _save_alignment_details_with_hdf5(self, alignment_details: Dict, dropped_counts: Dict, window_size: float, output_hdf5_path: str, warning_stats: Dict = None, original_data: Dict = None, original_mcap_file: str = None):
        """保存报警统计JSON文件到与HDF5文件相同的目录"""
        import json
        from datetime import datetime
        from pathlib import Path
        
        # 如果是backfill_on_grid或hybrid_alignment策略且有报警统计，只生成报警统计JSON文件
        if self.alignment_config.strategy in ['backfill_on_grid', 'hybrid_alignment'] and warning_stats:
            self._save_warning_stats_json(warning_stats, output_hdf5_path, original_data, original_mcap_file)
        else:
            # window_strict策略仍然保存详细的对齐详情文件
            self._save_detailed_alignment_report(alignment_details, dropped_counts, window_size, output_hdf5_path)
    
    def _save_warning_stats_json(self, warning_stats: Dict, output_hdf5_path: str, original_data: Dict = None, original_mcap_file: str = None):
        """保存报警统计信息（已弃用，现在直接保存到HDF5文件中）"""
        # 统计报警总数用于日志记录
        total_warnings = 0
        for topic_name, warnings in warning_stats.items():
            if warnings:
                total_warnings += len(warnings)
        
        log_info(f"Warning statistics saved to HDF5 file (total {total_warnings} warnings)", verbose_only=True)
    
    def _calculate_topic_end_analysis(self, original_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """计算话题结束时间分析"""
        try:
            # 过滤出有数据的话题
            topics_with_data = {topic: data for topic, data in original_data.items() if len(data) > 0}
            if not topics_with_data:
                return {}
            
            # 计算每个话题的开始和结束时间
            topic_times = {}
            for topic_name, data_list in topics_with_data.items():
                timestamps = [msg['timestamp'] for msg in data_list]
                topic_times[topic_name] = {
                    'start_time': min(timestamps),
                    'end_time': max(timestamps),
                    'duration': max(timestamps) - min(timestamps)
                }
            
            # 找到最早结束的话题
            earliest_end_topic = min(topic_times.keys(), key=lambda t: topic_times[t]['end_time'])
            earliest_end_time = topic_times[earliest_end_topic]['end_time']
            
            # 计算每个话题结束时间与最早结束时间的差值
            end_time_analysis = {}
            for topic_name, times in topic_times.items():
                end_time_diff = times['end_time'] - earliest_end_time
                duration_percentage = (end_time_diff / times['duration'] * 100) if times['duration'] > 0 else 0.0
                
                end_time_analysis[topic_name] = {
                    'start_time': times['start_time'],
                    'end_time': times['end_time'],
                    'duration': times['duration'],
                    'end_time_diff_from_earliest': end_time_diff,
                    'duration_percentage': f"{round(duration_percentage, 2)}%"
                }
            
            analysis_result = {
                'earliest_end_topic': earliest_end_topic,
                'earliest_end_time': earliest_end_time,
                'description': 'Topic end time analysis: shows the difference between each topic end time and the earliest end time, and its percentage in the topic duration',
                'topic_analysis': end_time_analysis
            }
            
            log_info(f"Topic end time analysis completed:", verbose_only=True)
            log_info(f"  Earliest end topic: {earliest_end_topic} (time: {earliest_end_time:.6f}s)", verbose_only=True)
            for topic_name, analysis in end_time_analysis.items():
                if topic_name != earliest_end_topic:
                    log_info(f"  {topic_name}: diff={analysis['end_time_diff_from_earliest']:+.6f}s, percentage={analysis['duration_percentage']}", verbose_only=True)
            
            return analysis_result
            
        except Exception as e:
            log_warning(f"Failed to calculate topic end time analysis: {e}", verbose_only=True)
            return {}
    
    def _save_detailed_alignment_report(self, alignment_details: Dict, dropped_counts: Dict, window_size: float, output_hdf5_path: str):
        """保存详细的对齐报告（用于非backfill_on_grid策略）"""
        import json
        from datetime import datetime
        from pathlib import Path
        
        # 创建对齐详情报告（添加中文注释）
        report = {
            'timestamp': datetime.now().isoformat(),
            'alignment_window': window_size,
            'hdf5_file': str(Path(output_hdf5_path).name),
            '注释': '本文件记录了时间戳对齐的详细差值信息，单位为秒',
            'topics': {}
        }
        
        for topic_name, details in alignment_details.items():
            if not details:
                continue
            
            # 计算统计信息
            time_diffs = [d['time_diff'] for d in details]
            abs_time_diffs = [d['abs_time_diff'] for d in details]
            
            topic_report = {
                'aligned_count': len(details),
                'dropped_count': dropped_counts[topic_name],
                '说明': 'aligned_count 为成功对齐的数据点数量，dropped_count 为因超出窗口被舍弃的数量',
                'statistics': {
                    'min_diff': float(min(time_diffs)),
                    'max_diff': float(max(time_diffs)),
                    'mean_diff': float(np.mean(time_diffs)),
                    'std_diff': float(np.std(time_diffs)),
                    'min_abs_diff': float(min(abs_time_diffs)),
                    'max_abs_diff': float(max(abs_time_diffs)),
                    'mean_abs_diff': float(np.mean(abs_time_diffs))
                },
                'statistics_注释': 'diff 为(选择时间-目标时间)，abs为绝对值；越小表示对齐越好',
                'alignment_details': details,
                'alignment_details_注释': '每个条目包含 target_time(目标时间), selected_time(被选中的数据时间), time_diff(差值), abs_time_diff(差值绝对值)'
            }
            
            report['topics'][topic_name] = topic_report
        
        # 与HDF5文件保存在同一目录
        hdf5_path = Path(output_hdf5_path)
        report_dir = hdf5_path.parent
        # 使用HDF5文件名作为前缀
        base_name = hdf5_path.stem
        filename = f"{base_name}_alignment_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = str(Path(report_dir) / filename)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            log_info(f"对齐详情已保存到文件: {output_file}", verbose_only=True)
        except Exception as e:
            log_warning(f"保存对齐详情文件失败: {e}", verbose_only=True)
    
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

def load_config(config_path: str) -> Tuple[List[TopicConfig], AlignmentConfig, Dict[str, Any]]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载话题配置
    topic_configs = []
    for topic_config in config.get('topics', []):
        topic_configs.append(TopicConfig(
            topic_name=topic_config['topic_name'],
            message_type=topic_config['message_type'],
            hdf5_path=topic_config['hdf5_path'],
            data_type=topic_config.get('data_type', 'float32'),
            description=topic_config.get('description', ''),
            custom_processor=topic_config.get('custom_processor'),
            custom_params=topic_config.get('custom_params', {}),
            # 新增字段
            topic_patterns=topic_config.get('topic_patterns'),
            auto_detect_type=topic_config.get('auto_detect_type', False),
            force_create_subdatasets=topic_config.get('force_create_subdatasets', False)
        ))
    
    # 加载对齐配置
    alignment_config = AlignmentConfig(
        main_timeline_topic=config.get('alignment', {}).get('main_timeline_topic'),
        alignment_window=config.get('alignment', {}).get('alignment_window', 0.05),
        target_fps=config.get('alignment', {}).get('target_fps', 30.0),
        sample_drop=config.get('alignment', {}).get('sample_drop', 0),
        relative_start=config.get('alignment', {}).get('relative_start', False),
        delta_action=config.get('alignment', {}).get('delta_action', False),
        strategy=config.get('alignment', {}).get('strategy', 'window_strict'),
        grid_fps=config.get('alignment', {}).get('grid_fps', 15.0),
        expand_start=config.get('alignment', {}).get('expand_start', False),
        # 插值配置
        interpolation_points=config.get('alignment', {}).get('interpolation_points', 2),
        interpolation_method=config.get('alignment', {}).get('interpolation_method', 'linear'),
        # 固定数据时长（秒），None表示不限制
        target_duration=(
            config.get('alignment', {}).get('target_duration')
            if config.get('alignment', {}).get('target_duration') is not None
            else config.get('alignment', {}).get('fixed_duration')
        ),
    )
    
    # 加载数据合并配置
    data_merging_config = config.get('data_merging', {})
    
    return topic_configs, alignment_config, data_merging_config

def convert_mcap_file(mcap_path: str, output_path: str, 
                     topic_configs: List[TopicConfig], 
                     alignment_config: AlignmentConfig,
                     data_merging_config: Dict[str, Any] = None) -> bool:
    """转换单个MCAP文件"""
    # 创建性能监控器
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # 创建读取器
        reader = FlexibleMcapReader(topic_configs)
        
        # 处理MCAP文件
        monitor.mark("mcap_reading")
        data = reader.process_mcap(mcap_path, monitor)
        if not data:
            # 失败日志已在process_mcap中保存
            return False
        
        # 计算总时间戳数量（用于对齐精度计算）
        total_timestamps = 0
        if alignment_config.strategy == 'backfill_on_grid' or alignment_config.strategy == 'hybrid_alignment':
            # 对于网格策略，需要计算网格点数
            topics_with_data = {t: msgs for t, msgs in data.items() if len(msgs) > 0}
            if topics_with_data:
                topic_starts = {t: min(m['timestamp'] for m in msgs) for t, msgs in topics_with_data.items()}
                topic_ends = {t: max(m['timestamp'] for m in msgs) for t, msgs in topics_with_data.items()}
                grid_start = max(topic_starts.values())
                grid_end = max(topic_ends.values())
                fps = alignment_config.grid_fps if alignment_config.grid_fps and alignment_config.grid_fps > 0 else 15.0
                if grid_end > grid_start:
                    total_timestamps = int((grid_end - grid_start) * fps) + 1
        else:
            # 对于window_strict策略，使用主时间线的话题数据量
            if alignment_config.main_timeline_topic and alignment_config.main_timeline_topic in data:
                total_timestamps = len(data[alignment_config.main_timeline_topic])
            else:
                # 自动选择数据量最小的话题
                valid_topics = {k: v for k, v in data.items() if len(v) > 0}
                if valid_topics:
                    main_topic = min(valid_topics.keys(), key=lambda k: len(valid_topics[k]))
                    total_timestamps = len(valid_topics[main_topic])
        
        # 时间戳对齐（根据策略选择）
        monitor.mark("timestamp_alignment")
        if alignment_config.strategy == 'backfill_on_grid':
            # 使用原始的等间隔网格回填策略
            aligned_data, alignment_details, dropped_counts, window_size, warning_stats = reader.align_data_backfill_on_grid(
                data, alignment_config)
        elif alignment_config.strategy == 'hybrid_alignment':
            # 使用混合对齐策略（图像回填+数值插值）
            aligned_data, alignment_details, dropped_counts, window_size, warning_stats = reader.align_data_hybrid(
                data, alignment_config)
        else:
            # 使用窗口严格对齐策略
            aligned_data, alignment_details, dropped_counts = reader.align_data_with_window(
                data, alignment_config)
            window_size = alignment_config.alignment_window
            warning_stats = {}
        
        # 转换到HDF5
        monitor.mark("hdf5_conversion")
        converter = FlexibleHdf5Converter(topic_configs, alignment_config, data_merging_config)
        success = converter.convert_to_hdf5(aligned_data, output_path, alignment_details, dropped_counts, window_size, warning_stats, data, mcap_path)
        
        if not success:
            # 保存HDF5转换失败日志
            _save_failure_log(mcap_path, "HDF5转换失败")
            return False
        
        # 计算对齐精度指标
        alignment_metrics = monitor.calculate_alignment_metrics(alignment_details, dropped_counts, total_timestamps)
        
        # 打印性能指标
        monitor.print_metrics(mcap_path, output_path, alignment_metrics)
        
        return success
        
    except Exception as e:
        logger.error(f"转换文件失败 {mcap_path}: {e}")
        # 保存异常失败日志
        _save_failure_log(mcap_path, str(e))
        return False

def main():
    """主函数"""
    global verbose_mode
    
    parser = argparse.ArgumentParser(description='灵活的MCAP到HDF5转换工具')
    parser.add_argument('--config', '-c', required=True, help='配置文件路径')
    parser.add_argument('--input', '-i', required=True, help='输入MCAP文件或目录')
    parser.add_argument('--output', '-o', required=True, help='输出HDF5文件或目录')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置全局详细模式
    verbose_mode = args.verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 加载配置
    try:
        topic_configs, alignment_config, data_merging_config = load_config(args.config)
        log_info(f"加载配置成功，包含 {len(topic_configs)} 个话题")
        if data_merging_config:
            log_info(f"数据合并配置: {list(data_merging_config.keys())}", verbose_only=True)
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return 1
    
    # 处理输入路径
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # 单个文件 - 直接使用episode_0命名
        if output_path.is_dir():
            final_file = output_path / "episode_0.hdf5"
        else:
            final_file = output_path.parent / "episode_0.hdf5"
        
        success = convert_mcap_file(str(input_path), str(final_file), 
                                  topic_configs, alignment_config, data_merging_config)
        if success:
            log_info(f"转换成功: {input_path} -> {final_file}")
        else:
            logger.error(f"转换失败: {input_path}")
            return 1
    
    elif input_path.is_dir():
        # 目录
        mcap_files = list(input_path.glob('*.mcap'))
        if not mcap_files:
            logger.error(f"在目录 {input_path} 中未找到MCAP文件")
            return 1
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        log_info(f"找到 {len(mcap_files)} 个MCAP文件")
        
        # 创建文件名映射文件
        filename_mapping = {}
        
        success_count = 0
        
        for i, mcap_file in enumerate(tqdm.tqdm(mcap_files, desc="转换进度", 
                                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
            # 先使用原始文件名转换
            original_output = output_path / f"{mcap_file.stem}.hdf5"
            
            if convert_mcap_file(str(mcap_file), str(original_output), 
                               topic_configs, alignment_config, data_merging_config):
                # 转换成功后立即重命名
                final_file = output_path / f"episode_{i}.hdf5"
                log_info(f"开始重命名: {original_output} -> {final_file}", verbose_only=True)
                try:
                    import shutil
                    shutil.move(str(original_output), str(final_file))
                    log_info(f"HDF5文件重命名成功: {original_output.name} -> {final_file.name}", verbose_only=True)
                    
                    # 重命名对应的报警日志文件
                    _rename_warning_logs(original_output, final_file)
                    
                    # 记录文件名映射
                    filename_mapping[f"episode_{i}"] = {
                        "original_file": mcap_file.name,
                        "output_file": final_file.name
                    }
                    
                    success_count += 1
                    log_info(f"✓ 转换成功: {mcap_file.name} -> {final_file.name}")
                except Exception as e:
                    logger.error(f"重命名文件失败 {original_output}: {e}")
                    import traceback
                    log_debug(f"重命名错误详情: {traceback.format_exc()}")
                    # 如果重命名失败，仍然记录映射关系
                    filename_mapping[f"episode_{i}"] = {
                        "original_file": mcap_file.name,
                        "output_file": original_output.name
                    }
                    success_count += 1
                    log_info(f"✓ 转换成功: {mcap_file.name} -> {original_output.name}")
            else:
                logger.error(f"转换失败: {mcap_file.name}")
        
        # 保存文件名映射
        mapping_file = output_path / "filename_mapping.json"
        try:
            import json
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(filename_mapping, f, indent=2, ensure_ascii=False)
            log_info(f"文件名映射已保存: {mapping_file}", verbose_only=True)
        except Exception as e:
            log_warning(f"保存文件名映射失败: {e}", verbose_only=True)
        
        log_info(f"转换完成: {success_count}/{len(mcap_files)} 个文件成功")
    
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return 1
    
    return 0

def _rename_warning_logs(original_file: Path, final_file: Path):
    """重命名报警日志文件"""
    try:
        import glob
        from pathlib import Path
        
        # 查找与原始文件相关的报警日志文件
        original_stem = original_file.stem
        final_stem = final_file.stem
        
        # 查找所有可能的报警日志文件
        warning_pattern = f"{original_stem}_warning_stats_*.json"
        warning_files = list(original_file.parent.glob(warning_pattern))
        
        for warning_file in warning_files:
            # 生成新的报警日志文件名
            new_warning_file = warning_file.parent / f"{final_stem}_warning_stats_{warning_file.name.split('_warning_stats_')[1]}"
            
            try:
                import shutil
                shutil.move(str(warning_file), str(new_warning_file))
                log_info(f"报警日志重命名成功: {warning_file.name} -> {new_warning_file.name}", verbose_only=True)
            except Exception as e:
                log_warning(f"重命名报警日志失败 {warning_file}: {e}", verbose_only=True)
                
    except Exception as e:
        log_warning(f"重命名报警日志时出错: {e}", verbose_only=True)

def _save_failure_log(mcap_file: str, reason: str):
    """保存转换失败日志（全局函数）"""
    try:
        from datetime import datetime
        import json
        from pathlib import Path
        
        # 确定输出目录（与MCAP文件同目录）
        mcap_path = Path(mcap_file)
        report_dir = mcap_path.parent
        
        # 生成失败日志文件名
        base_name = mcap_path.stem
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"{base_name}_failure_{ts}.json"
        
        payload = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'mcap_file': str(mcap_path.name),
            'reason': reason,
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log_info(f"转换失败日志已保存: {report_file}", verbose_only=True)
    except Exception as e:
        log_warning(f"保存转换失败日志时出错: {e}", verbose_only=True)

if __name__ == "__main__":
    sys.exit(main())
