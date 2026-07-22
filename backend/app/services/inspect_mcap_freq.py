#!/usr/bin/env python3

import sys
import argparse
import logging
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from collections import defaultdict
import numpy as np
import os

# Configure logging to a file that we can check
logging.basicConfig(
    filename='/tmp/mcap_analysis.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_frequency_stats(mcap_path):
    """
    Analyze MCAP file to get frequency statistics for each topic.
    Returns a list of dictionaries containing topic stats.
    """
    logger.info(f"Starting analysis for: {mcap_path}")
    
    if not os.path.exists(mcap_path):
        logger.error(f"File not found: {mcap_path}")
        return []

    # Handle directory path
    real_path = mcap_path
    if os.path.isdir(mcap_path):
        logger.info(f"Path is a directory, searching for .mcap files in {mcap_path}")
        mcap_files = [f for f in os.listdir(mcap_path) if f.endswith('.mcap')]
        if not mcap_files:
            logger.error(f"No .mcap files found in directory: {mcap_path}")
            return []
        # Sort to find the first one (e.g. _0.mcap)
        mcap_files.sort()
        real_path = os.path.join(mcap_path, mcap_files[0])
        logger.info(f"Selected MCAP file: {real_path}")

    topic_timestamps = defaultdict(list)
    topic_counts = defaultdict(int)
    results = []
    
    try:
        with open(real_path, "rb") as f:
            logger.info("File opened successfully, creating reader...")
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            # 统计所有消息
            msg_count = 0
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                # 使用 log_time 作为收到消息的时间，单位为纳秒
                timestamp = message.log_time
                
                topic_timestamps[topic].append(timestamp)
                topic_counts[topic] += 1
                msg_count += 1
            
            logger.info(f"Read {msg_count} messages across {len(topic_timestamps)} topics")
                
        for topic in sorted(topic_timestamps.keys()):
            timestamps = sorted(topic_timestamps[topic])
            count = len(timestamps)
            
            stats = {
                "topic": topic,
                "count": count,
                "frequency": 0.0,
                "period_ms": 0.0,
                "min_delta_ms": 0.0,
                "max_delta_ms": 0.0
            }
            
            if count >= 2:
                # 计算时间差 (纳秒 -> 秒)
                timestamps_sec = np.array(timestamps) / 1e9
                diffs = np.diff(timestamps_sec)
                
                if len(diffs) > 0:
                    duration = timestamps_sec[-1] - timestamps_sec[0]
                    
                    if duration > 0:
                        freq = (count - 1) / duration
                        stats["frequency"] = freq
                        stats["period_ms"] = (1.0 / freq) * 1000 if freq > 0 else 0
                    
                    stats["min_delta_ms"] = np.min(diffs) * 1000
                    stats["max_delta_ms"] = np.max(diffs) * 1000
            
            results.append(stats)
        
        logger.info(f"Analysis completed. Found {len(results)} topics.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing file: {e}", exc_info=True)
        print(f"Error analyzing file: {e}")
        return []

def analyze_frequency(mcap_path):
    print(f"Analyzing {mcap_path}...")
    
    results = get_frequency_stats(mcap_path)
    
    print("\n" + "="*100)
    print(f"{'Topic':<40} | {'Count':<8} | {'Freq (Hz)':<10} | {'Period (ms)':<12} | {'Min/Max Δt (ms)':<20}")
    print("-" * 100)
    
    for stats in results:
        topic = stats['topic']
        count = stats['count']
        
        if count < 2:
            print(f"{topic:<40} | {count:<8} | {'N/A':<10} | {'N/A':<12} | {'N/A':<20}")
            continue
            
        freq = stats['frequency']
        period_ms = stats['period_ms']
        min_dt_ms = stats['min_delta_ms']
        max_dt_ms = stats['max_delta_ms']
        
        print(f"{topic:<40} | {count:<8} | {freq:<10.2f} | {period_ms:<12.2f} | {min_dt_ms:.1f}/{max_dt_ms:.1f}")
            
    print("="*100 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MCAP topic frequency")
    parser.add_argument("mcap_file", help="Path to MCAP file")
    args = parser.parse_args()
    
    analyze_frequency(args.mcap_file)
