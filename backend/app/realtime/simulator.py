import asyncio
from typing import Dict
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.crud.job import update_job
from app.realtime.job_ws import manager


class JobSimulator:
    """作业模拟器（模拟采集进度）"""
    
    def __init__(self):
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_simulation(
        self,
        db: AsyncSession,
        job_id: UUID,
        duration_sec: int = 30
    ):
        """启动模拟采集任务"""
        job_id_str = str(job_id)
        
        # 如果已有任务在运行，先取消
        if job_id_str in self.running_tasks:
            self.running_tasks[job_id_str].cancel()
        
        # 更新作业状态为 RUNNING
        from app.schemas.job import JobUpdate
        await update_job(
            db,
            job_id,
            JobUpdate(
                status="RUNNING",
                progress=0,
                started_at=datetime.utcnow().isoformat()
            )
        )
        
        # 广播初始状态
        await manager.broadcast_progress(job_id_str, "RUNNING", 0)
        
        # 创建后台任务
        task = asyncio.create_task(
            self._simulate_progress(db, job_id, duration_sec)
        )
        self.running_tasks[job_id_str] = task
    
    async def _simulate_progress(
        self,
        db: AsyncSession,
        job_id: UUID,
        duration_sec: int
    ):
        """模拟进度更新"""
        job_id_str = str(job_id)
        total_steps = 50  # 50 步到 100%
        step_delay = duration_sec / total_steps  # 每步延迟
        
        try:
            for step in range(1, total_steps + 1):
                await asyncio.sleep(step_delay)
                
                progress = min(step * 2, 100)  # 每次 +2，最多 100
                
                # 更新数据库
                from app.schemas.job import JobUpdate
                await update_job(
                    db,
                    job_id,
                    JobUpdate(progress=progress)
                )
                
                # 广播进度
                await manager.broadcast_progress(job_id_str, "RUNNING", progress)
            
            # 完成
            from app.schemas.job import JobUpdate
            await update_job(
                db,
                job_id,
                JobUpdate(
                    status="SUCCEEDED",
                    progress=100,
                    finished_at=datetime.utcnow().isoformat(),
                    mcap_path=f"/data/daq/outputs/{job_id_str}.mcap",
                    mcap_size_bytes=1024 * 1024 * 100,  # 假数据：100MB
                    duration_sec=duration_sec
                )
            )
            
            await manager.broadcast_progress(job_id_str, "SUCCEEDED", 100)
            
        except asyncio.CancelledError:
            # 被取消，更新状态为 CANCELED
            from app.schemas.job import JobUpdate
            await update_job(
                db,
                job_id,
                JobUpdate(
                    status="CANCELED",
                    finished_at=datetime.utcnow().isoformat()
                )
            )
            await manager.broadcast_progress(job_id_str, "CANCELED", 0)
        finally:
            # 清理任务
            if job_id_str in self.running_tasks:
                del self.running_tasks[job_id_str]
    
    def cancel_simulation(self, job_id: UUID):
        """取消模拟任务"""
        job_id_str = str(job_id)
        if job_id_str in self.running_tasks:
            self.running_tasks[job_id_str].cancel()


simulator = JobSimulator()

