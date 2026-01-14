from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import uuid
import logging

from src.services.websocket_manager import manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/vessels")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time vessel updates.

    Client messages:
    - {"type": "subscribe", "vessel_ids": ["lng-001", "lng-002"]}
      (empty array = subscribe to all vessels)

    Server messages:
    - {"type": "vessel_update", "data": {...}}
    - {"type": "prediction_created", "data": {...}}
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "subscribe":
                vessel_ids = data.get("vessel_ids", [])
                await manager.subscribe(client_id, vessel_ids)

                # Send confirmation
                await manager.send_to_client(client_id, {
                    "type": "subscribed",
                    "vessel_count": len(vessel_ids) if vessel_ids else "all"
                })

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.disconnect(client_id)
