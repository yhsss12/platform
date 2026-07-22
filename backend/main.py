# 这个文件已废弃，请使用 app.main:app
# 为了兼容性，这里重定向到 app.main
import sys
from pathlib import Path

# 添加 app 目录到路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# 导入新的主应用
from app.main import app

