
import asyncio
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime

# Add backend directory to path
BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import AsyncSessionLocal
from app.crud.job import create_job, update_job, get_job
from app.schemas.job import JobCreate, JobUpdate
from app.models.job import JobStatus
from app.models.user import User, UserRole

# Mock user
mock_user = User(
    id=str(uuid4()),
    account_id="test_user",
    username="test_user",
    password_hash="hashed_password",
    role=UserRole.ADMIN,
    is_active=True,
)

async def test_backend_logic():
    print("Starting backend logic test...")
    
    async with AsyncSessionLocal() as db:
        # 1. Create a job
        task_id = uuid4()
        job_in = JobCreate(
            task_id=task_id,
            operator_name="tester",
            status=JobStatus.PENDING,
            collection_quantity=1,
            completed_count=0
        )
        
        print(f"Creating job with collection_quantity={job_in.collection_quantity}...")
        job = await create_job(db, job_in)
        print(f"Job created: ID={job.id}, Status={job.status}, Progress={job.completed_count}/{job.collection_quantity}")
        
        # 2. Simulate frontend update payload
        # Frontend sends: status="COMPLETED", progress={current: 1, total: 1}
        print("\nSimulating frontend update...")
        
        # In routes_jobs.py, the payload is converted to JobUpdate
        # progress is a dict in the raw JSON, but JobUpdate model might handle it differently
        # Let's manually construct the JobUpdate as the route would
        
        update_data = {
            "status": "COMPLETED",
            "progress": {"current": 1, "total": 1}
        }
        
        # Manually handle progress dict logic as in routes_jobs.py
        job_update = JobUpdate(**update_data)
        
        # Logic from routes_jobs.py
        if isinstance(job_update.progress, dict):
            current = job_update.progress.get("current")
            total = job_update.progress.get("total")
            
            print(f"Processing progress dict: current={current}, total={total}")
            
            if current is not None:
                job_update.completed_count = current
            if total is not None:
                job_update.collection_quantity = total
                
            # Manually add to set (crucial step in routes_jobs.py)
            if hasattr(job_update, "model_fields_set"):
                job_update.model_fields_set.add("completed_count")
                job_update.model_fields_set.add("collection_quantity")
            
            # Update percentage progress (0-100) - MISSING IN PREVIOUS VERSION
            c = current if current is not None else (job_update.completed_count or 0)
            t = total if total is not None else (job_update.collection_quantity or 0)
            
            if t > 0:
                job_update.progress = int((c / t) * 100)
            else:
                job_update.progress = 0
                
            print(f"Calculated progress percent: {job_update.progress}")
        
        print(f"JobUpdate prepared: status={job_update.status}, completed_count={job_update.completed_count}")
        
        # 3. Apply update
        updated_job = await update_job(db, job.id, job_update)
        print(f"Job updated: Status={updated_job.status}, Progress={updated_job.completed_count}/{updated_job.collection_quantity}")
        
        # 4. Verify
        if updated_job.status == JobStatus.COMPLETED:
            print("✅ SUCCESS: Job status is COMPLETED")
        else:
            print(f"❌ FAILURE: Job status is {updated_job.status}")
            
        # Clean up
        # await delete_job(db, job.id)

if __name__ == "__main__":
    asyncio.run(test_backend_logic())
