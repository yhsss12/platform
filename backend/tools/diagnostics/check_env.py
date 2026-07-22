#!/usr/bin/env python3
"""
检查环境变量是否正确加载
"""
import os
import sys
from pathlib import Path

# 模拟 main.py 的加载逻辑
try:
    from dotenv import load_dotenv
    
    backend_dir = Path(__file__).resolve().parents[2]  # backend/
    project_root = backend_dir.parent  # 项目根目录
    
    # 优先加载 backend/.env，如果不存在则加载项目根目录的 .env
    env_paths = [
        backend_dir / ".env",  # backend/.env
        project_root / ".env",  # 项目根目录/.env
    ]
    
    loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            print(f"✓ 已加载环境变量文件: {env_path}")
            loaded = True
            break
    
    if not loaded:
        print("⚠ 警告: 未找到 .env 文件")
        print(f"   查找位置: {env_paths[0]}, {env_paths[1]}")
except Exception as e:
    print(f"⚠ 警告: 自动加载 .env 失败: {e}")

# 检查环境变量
print("\n环境变量状态:")
print(f"OPENAI_API_KEY: {'✓ 已设置' if os.getenv('OPENAI_API_KEY') else '✗ 未设置'}")
print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', '✗ 未设置')}")
print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', '✗ 未设置')}")
print(f"HDF5_DATA_DIR: {os.getenv('HDF5_DATA_DIR', '✗ 未设置')}")

# 检查 label_task_description.py 是否能导入
print("\n检查 label_task_description 导入:")
try:
    sys.path.insert(0, str(project_root))
    from label_task_description import gen_task_description
    print("✓ label_task_description 导入成功")
except Exception as e:
    print(f"✗ label_task_description 导入失败: {e}")






















