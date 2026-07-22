"""调试：打印当前数据资产路径与数据库模式（仅 PostgreSQL，无本地 .db 文件）。"""
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))
try:
    from app.db.data_assets_session import DATA_ASSETS_ROOT
    from app.core.config import settings
    print("数据库: PostgreSQL (统一库)")
    print(f"DATABASE_URL: {settings.DATABASE_URL[:50]}...")
    print(f"DATA_ASSETS_ROOT (资产文件目录): {DATA_ASSETS_ROOT}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
