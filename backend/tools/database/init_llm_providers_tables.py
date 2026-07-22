#!/usr/bin/env python3
"""
已废弃：项目仅使用 PostgreSQL，厂商/模型表由应用迁移或 create_all 创建。执行即退出。
"""
import sys

if __name__ == "__main__":
    print("已废弃：请使用 PostgreSQL 与主应用迁移/启动逻辑创建 LLM 相关表。")
    sys.exit(0)
