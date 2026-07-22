from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.services.agent_stream_proxy import (
    list_streams_via_agent,
    get_stream_generator_via_tunnel_or_agent,
    get_stream_status_via_tunnel,
)

router = APIRouter()

@router.get("/list")
async def list_cameras(
    device_id: int | None = Query(
        default=None,
        description="设备 ID，可用于根据设备选择对应采集端 Agent",
    ),
    agent_id: str | None = Query(
        default=None,
        description="显式指定采集端 Agent ID，优先级高于 device_id",
    ),
):
    """
    Get list of available cameras.
    Returns: [{"id": "camera1", "name": "Camera 1"}, ...]
    """
    return await list_streams_via_agent(device_id=device_id, agent_id=agent_id)


@router.get("/status")
async def get_stream_status(
    device_id: int | None = Query(
        default=None,
        description="设备 ID，可用于根据设备选择对应采集端 Agent",
    ),
    agent_id: str | None = Query(
        default=None,
        description="显式指定采集端 Agent ID，优先级高于 device_id",
    ),
):
    return await get_stream_status_via_tunnel(device_id=device_id, agent_id=agent_id)


@router.get("/{camera_id}")
async def get_camera_stream(
    camera_id: str,
    device_id: int | None = Query(
        default=None,
        description="设备 ID，可用于根据设备选择对应采集端 Agent",
    ),
    agent_id: str | None = Query(
        default=None,
        description="显式指定采集端 Agent ID，优先级高于 device_id",
    ),
):
    """
    Get MJPEG stream for a specific camera.
    Supported IDs: camera1, camera2, camera3, camera4
    """
    return StreamingResponse(
        get_stream_generator_via_tunnel_or_agent(
            camera_id,
            device_id=device_id,
            agent_id=agent_id,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
