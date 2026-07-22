from typing import Dict, Set, Any, Optional
from fastapi import WebSocket
import json
from uuid import UUID


class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        # job_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """接受 WebSocket 连接"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """断开 WebSocket 连接"""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if len(self.active_connections[job_id]) == 0:
                del self.active_connections[job_id]
    
    async def broadcast_log(self, job_id: str, message: str):
        """广播日志"""
        if job_id not in self.active_connections:
            return

        payload = {
            "type": "log",
            "jobId": job_id,
            "message": message,
        }
        disconnected = set()
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(payload)
            except Exception:
                disconnected.add(websocket)
        for ws in disconnected:
            self.disconnect(ws, job_id)

    async def broadcast_progress(self, job_id: str, status: Any, progress: Optional[int] = None):
        """广播进度更新"""
        if job_id not in self.active_connections:
            return

        if isinstance(status, dict) and progress is None:
            message = dict(status)
            message.setdefault("type", "progress")
            message.setdefault("jobId", job_id)
        else:
            message = {
                "type": "progress",
                "jobId": job_id,
                "status": status,
                "progress": int(progress or 0),
            }
        
        disconnected = set()
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # 清理断开的连接
        for ws in disconnected:
            self.disconnect(ws, job_id)


manager = ConnectionManager()

