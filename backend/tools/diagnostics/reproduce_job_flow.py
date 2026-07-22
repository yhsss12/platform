import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import BackgroundTasks

# Add backend directory to path
BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from app.api.routes_jobs import create_new_job, read_jobs, update_job_by_id
from app.db.data_assets_session import DataAssetsSessionLocal
from app.models.user import User, UserRole
from app.schemas.job import JobCreate, JobUpdate


async def run_test():
    print("Starting test...")
    mock_user = User(
        id=str(uuid4()),
        account_id="test_user",
        username="test_user",
        password_hash="hash",
        role=UserRole.USER,
    )
    mock_request = MagicMock()

    async with DataAssetsSessionLocal() as db:
        try:
            task_id = uuid4()
            job_in = JobCreate(
                task_id=task_id,
                operator_name="tester",
                collection_quantity=1,
                status="PENDING",
            )
            print(f"Creating job for task {task_id}...")
            response = await create_new_job(job_in, mock_request, BackgroundTasks(), db, mock_user)
            if not response.ok or not response.data:
                print("Failed to create job")
                return

            job = response.data
            job_id = job.id
            print(f"Job created: {job_id}, status={job.status}, completed={job.completed_count}")

            update_payload = JobUpdate(
                status="COMPLETED",
                progress={"current": 1, "total": 1},
                mcap_path="/tmp/test.mcap",
                register_collect_asset=True,
                mcap_size_bytes=1024,
                duration_sec=10,
            )
            print("Updating job...")
            update_response = await update_job_by_id(
                job_id, update_payload, mock_request, BackgroundTasks(), db, mock_user
            )
            if not update_response.ok:
                print(f"Update failed: {update_response.error}")
                return

            updated_job = update_response.data
            print(
                f"Update response: status={updated_job.status}, "
                f"completed={updated_job.completed_count}, progress={updated_job.progress}"
            )

            print("Reading jobs list...")
            list_response = await read_jobs(task_id=task_id, db=db, current_user=mock_user)
            if not list_response.ok:
                print("Failed to list jobs")
                return

            found_job = next((j for j in list_response.data if j.id == job_id), None)
            if found_job:
                print(
                    f"List response: status={found_job.status}, "
                    f"completed={found_job.completed_count}, progress={found_job.progress}"
                )
                if found_job.status != "COMPLETED" or found_job.completed_count != 1:
                    print("❌ TEST FAILED: Data mismatch!")
                else:
                    print("✅ TEST PASSED: Data consistent.")
            else:
                print("Job not found in list")

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())
