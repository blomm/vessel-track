from fastapi import WebSocket
from typing import Dict, Set
import json
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of vessel_ids

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"WebSocket client {client_id} connected")

    async def disconnect(self, client_id: str):
        """Remove connection and subscriptions"""
        self.active_connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)
        logger.info(f"WebSocket client {client_id} disconnected")

    async def subscribe(self, client_id: str, vessel_ids: list):
        """Subscribe client to specific vessels (empty = all vessels)"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] = set(vessel_ids) if vessel_ids else set()
            logger.debug(f"Client {client_id} subscribed to {len(vessel_ids or [])} vessels")

    async def send_to_client(self, client_id: str, message: dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast_vessel_update(self, vessel_dict: dict):
        """Broadcast vessel update to subscribed clients"""
        vessel_id = vessel_dict.get('id')
        message = {
            "type": "vessel_update",
            "data": vessel_dict
        }

        for client_id, subscribed_vessels in self.subscriptions.items():
            # Send if subscribed to this vessel or subscribed to all (empty set)
            if not subscribed_vessels or vessel_id in subscribed_vessels:
                await self.send_to_client(client_id, message)

    async def broadcast_prediction(self, prediction_dict: dict):
        """Broadcast new prediction to all clients"""
        message = {
            "type": "prediction_created",
            "data": prediction_dict
        }

        for client_id in self.active_connections.keys():
            await self.send_to_client(client_id, message)


# Global manager instance
manager = WebSocketManager()
