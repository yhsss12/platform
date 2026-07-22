"""兼容入口：委托 app.workers.artifact_upload_worker。"""

from app.workers.artifact_upload_worker import main, run_loop, run_once

__all__ = ["main", "run_once", "run_loop"]

if __name__ == "__main__":
    main()
