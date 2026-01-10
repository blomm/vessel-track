# Phase 7: Kafka Event-Driven Integration

**Duration**: Days 29-33
**Goal**: Migrate from synchronous to event-driven architecture using Apache Kafka for scalable, asynchronous prediction processing

---

## Overview

This phase transforms the Vessel Track system from synchronous request-response to event-driven architecture. Key changes:

- **API Response Time**: 5-10s → <50ms (predictions processed asynchronously)
- **Scalability**: Horizontal scaling of prediction workers
- **Resilience**: Automatic retries, dead letter queues
- **Observability**: Complete event audit trail
- **Decoupling**: Services communicate via events

### Architecture Transformation

**Before (Synchronous)**:
```
API Request → Prediction Engine → RAG → AI (GPT-4o) → DB → WebSocket → Response
[5-10 second blocking operation]
```

**After (Event-Driven)**:
```
API Request → Publish Event → Response (50ms)
    ↓
[Async Pipeline]
Event → Worker → Prediction Engine → RAG → AI → Publish Result
    ↓
Result → [DB Writer | WebSocket | Slack | Learning] (parallel consumers)
```

---

## 7.1. Infrastructure Setup

### Add Kafka to Docker Compose

Docker Compose has been updated with Kafka and Zookeeper services.

**Verify** ([docker-compose.yml](../../service/docker-compose.yml:35-76)):
```yaml
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: vessel_track_zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
      - zookeeper_logs:/var/lib/zookeeper/log
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "2181"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: vessel_track_kafka
    depends_on:
      zookeeper:
        condition: service_healthy
    ports:
      - "9092:9092"
      - "29092:29092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
    volumes:
      - kafka_data:/var/lib/kafka/data
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 10s
      timeout: 10s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  zookeeper_data:
  zookeeper_logs:
  kafka_data:
```

**Start Infrastructure**:
```bash
cd service
docker-compose up -d

# Verify Kafka is running
docker-compose ps
# Should show: vessel_track_kafka (healthy)
# Should show: vessel_track_zookeeper (healthy)

# Check Kafka logs
docker-compose logs kafka
# Should see: "Kafka Server started"
```

### Install Kafka Python Client

Add to `service/pyproject.toml`:
```toml
[tool.poetry.dependencies]
confluent-kafka = "^2.3.0"
```

Install:
```bash
cd service
poetry add confluent-kafka
poetry install
```

---

## 7.2. Create Kafka Topics

### Create `service/scripts/create_kafka_topics.sh`:

```bash
#!/bin/bash

# Create Kafka topics with optimal configuration

KAFKA_BROKER="localhost:9092"

echo "Creating Kafka topics for Vessel Track..."

# Topic 1: vessel-position-updates
docker exec vessel_track_kafka kafka-topics \
  --bootstrap-server $KAFKA_BROKER \
  --create \
  --topic vessel-position-updates \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

echo "✓ Created vessel-position-updates"

# Topic 2: prediction-requests
docker exec vessel_track_kafka kafka-topics \
  --bootstrap-server $KAFKA_BROKER \
  --create \
  --topic prediction-requests \
  --partitions 5 \
  --replication-factor 1 \
  --config retention.ms=86400000 \
  --config compression.type=gzip \
  --if-not-exists

echo "✓ Created prediction-requests"

# Topic 3: prediction-results
docker exec vessel_track_kafka kafka-topics \
  --bootstrap-server $KAFKA_BROKER \
  --create \
  --topic prediction-results \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

echo "✓ Created prediction-results"

# List all topics
echo ""
echo "All topics:"
docker exec vessel_track_kafka kafka-topics \
  --bootstrap-server $KAFKA_BROKER \
  --list

echo ""
echo "Topics created successfully!"
```

**Run Script**:
```bash
cd service
chmod +x scripts/create_kafka_topics.sh
./scripts/create_kafka_topics.sh
```

**Verify Topics**:
```bash
docker exec vessel_track_kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --describe
```

---

## 7.3. Event Schemas

### Create `service/src/schemas/events.py`:

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional

# ============================================================================
# Vessel Position Update Event
# ============================================================================

class VesselPositionData(BaseModel):
    """Vessel position and state data"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    speed: float = Field(..., ge=0, description="Speed in knots")
    heading: float = Field(..., ge=0, lt=360, description="Heading in degrees")
    status: str = Field(..., description="Vessel status (underway, moored, etc)")

class VesselPositionUpdateEvent(BaseModel):
    """Event published when vessel position is updated"""
    event_id: str = Field(..., description="Unique event ID (UUID)")
    event_type: Literal["vessel.position.updated"] = "vessel.position.updated"
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    vessel_id: str = Field(..., description="Vessel ID")
    data: VesselPositionData
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "550e8400-e29b-41d4-a716-446655440000",
                "event_type": "vessel.position.updated",
                "timestamp": "2026-01-10T15:30:00.000Z",
                "vessel_id": "lng-001",
                "data": {
                    "lat": 29.7604,
                    "lon": -95.3698,
                    "speed": 12.5,
                    "heading": 275.0,
                    "status": "underway"
                },
                "metadata": {
                    "source": "api_update",
                    "correlation_id": "corr-abc123"
                }
            }
        }

# ============================================================================
# Prediction Request Event
# ============================================================================

class VesselSnapshot(BaseModel):
    """Snapshot of vessel state at prediction time"""
    lat: float
    lon: float
    speed: float
    heading: float
    status: str

class PredictionRequestData(BaseModel):
    """Data for prediction analysis request"""
    vessel_snapshot: VesselSnapshot
    terminal_filter: Optional[int] = Field(None, description="Analyze specific terminal only")
    priority: Literal["low", "normal", "high"] = Field("normal", description="Processing priority")

class PredictionRequestEvent(BaseModel):
    """Event to trigger prediction analysis"""
    event_id: str
    event_type: Literal["prediction.analysis.requested"] = "prediction.analysis.requested"
    timestamp: datetime
    vessel_id: str
    request_id: str = Field(..., description="Unique request ID for tracking")
    data: PredictionRequestData
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "660e8400-e29b-41d4-a716-446655440001",
                "event_type": "prediction.analysis.requested",
                "timestamp": "2026-01-10T15:30:05.000Z",
                "vessel_id": "lng-001",
                "request_id": "req-12345",
                "data": {
                    "vessel_snapshot": {
                        "lat": 29.7604,
                        "lon": -95.3698,
                        "speed": 12.5,
                        "heading": 275.0,
                        "status": "underway"
                    },
                    "terminal_filter": None,
                    "priority": "normal"
                },
                "metadata": {
                    "triggered_by": "position_update",
                    "correlation_id": "corr-abc123"
                }
            }
        }

# ============================================================================
# Prediction Result Event
# ============================================================================

class PredictionScores(BaseModel):
    """Breakdown of prediction confidence components"""
    proximity: float = Field(..., description="Proximity score (0-1)")
    speed: float = Field(..., description="Speed score (0-1)")
    heading: float = Field(..., description="Heading score (0-1)")
    rag_boost: float = Field(..., description="RAG historical boost (0-0.3)")
    ai_adjustment: float = Field(..., description="AI adjustment (-0.3 to +0.3)")

class PredictionResultData(BaseModel):
    """Complete prediction result data"""
    prediction_id: str = Field(..., description="Database prediction ID")
    terminal_id: int
    terminal_name: str
    confidence_score: float = Field(..., ge=0, le=1, description="Final confidence (0-1)")
    scores: PredictionScores
    distance_km: float = Field(..., description="Distance to terminal in km")
    eta_hours: Optional[float] = Field(None, description="Estimated time of arrival in hours")
    ai_reasoning: str = Field(..., description="Natural language explanation from AI")
    vessel_snapshot: VesselSnapshot

class PredictionResultEvent(BaseModel):
    """Event published when prediction analysis is complete"""
    event_id: str
    event_type: Literal["prediction.analysis.completed"] = "prediction.analysis.completed"
    timestamp: datetime
    vessel_id: str
    request_id: str = Field(..., description="Original request ID for correlation")
    data: PredictionResultData
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "770e8400-e29b-41d4-a716-446655440002",
                "event_type": "prediction.analysis.completed",
                "timestamp": "2026-01-10T15:30:10.000Z",
                "vessel_id": "lng-001",
                "request_id": "req-12345",
                "data": {
                    "prediction_id": "pred-78901",
                    "terminal_id": 1,
                    "terminal_name": "Sabine Pass LNG",
                    "confidence_score": 0.87,
                    "scores": {
                        "proximity": 0.75,
                        "speed": 1.0,
                        "heading": 0.82,
                        "rag_boost": 0.15,
                        "ai_adjustment": 0.05
                    },
                    "distance_km": 45.2,
                    "eta_hours": 3.6,
                    "ai_reasoning": "Vessel maintaining optimal approach speed...",
                    "vessel_snapshot": {
                        "lat": 29.7604,
                        "lon": -95.3698,
                        "speed": 12.5,
                        "heading": 275.0,
                        "status": "underway"
                    }
                },
                "metadata": {
                    "processing_time_ms": 2340,
                    "correlation_id": "corr-abc123"
                }
            }
        }
```

---

## 7.4. Kafka Producer Service

### Create `service/src/services/kafka_producer.py`:

```python
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import json
import logging
from typing import Optional, Callable
from datetime import datetime
import uuid

from src.config import settings
from src.schemas.events import (
    VesselPositionUpdateEvent,
    PredictionRequestEvent,
    PredictionResultEvent
)

logger = logging.getLogger(__name__)


class VesselEventProducer:
    """
    Kafka event producer for Vessel Track events.
    Publishes events to Kafka topics with proper serialization and error handling.
    """

    def __init__(self):
        self.producer = Producer({
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'compression.type': settings.KAFKA_COMPRESSION_TYPE,
            'batch.size': settings.KAFKA_BATCH_SIZE,
            'linger.ms': settings.KAFKA_LINGER_MS,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
        })
        logger.info(f"Kafka producer initialized: {settings.KAFKA_BOOTSTRAP_SERVERS}")

    def _delivery_callback(self, err, msg):
        """Callback for message delivery reports"""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} "
                f"[partition {msg.partition()}] at offset {msg.offset()}"
            )

    def _publish(
        self,
        topic: str,
        key: str,
        event: dict,
        callback: Optional[Callable] = None
    ):
        """
        Internal method to publish event to Kafka.

        Args:
            topic: Kafka topic name
            key: Partition key (usually vessel_id)
            event: Event dictionary
            callback: Optional delivery callback
        """
        try:
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8'),
                value=json.dumps(event).encode('utf-8'),
                callback=callback or self._delivery_callback
            )
            self.producer.poll(0)  # Trigger callbacks
            logger.debug(f"Published event to {topic}: {event['event_id']}")
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            raise

    def publish_vessel_position_update(
        self,
        vessel_id: str,
        lat: float,
        lon: float,
        speed: float,
        heading: float,
        status: str = "underway",
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Publish vessel position update event.

        Args:
            vessel_id: Vessel identifier
            lat: Latitude
            lon: Longitude
            speed: Speed in knots
            heading: Heading in degrees
            status: Vessel status
            correlation_id: Optional correlation ID

        Returns:
            event_id: Generated event ID
        """
        event_id = str(uuid.uuid4())

        event = VesselPositionUpdateEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            vessel_id=vessel_id,
            data={
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "heading": heading,
                "status": status
            },
            metadata={
                "source": "api_update",
                "correlation_id": correlation_id
            }
        ).model_dump(mode='json')

        self._publish(
            topic=settings.KAFKA_VESSEL_POSITIONS_TOPIC,
            key=vessel_id,
            event=event
        )

        logger.info(f"Published position update for vessel {vessel_id}")
        return event_id

    def publish_prediction_request(
        self,
        vessel_id: str,
        vessel_snapshot: dict,
        request_id: Optional[str] = None,
        terminal_filter: Optional[int] = None,
        priority: str = "normal",
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Publish prediction analysis request event.

        Args:
            vessel_id: Vessel identifier
            vessel_snapshot: Current vessel state snapshot
            request_id: Optional request ID (generated if not provided)
            terminal_filter: Optional specific terminal to analyze
            priority: Processing priority (low/normal/high)
            correlation_id: Optional correlation ID

        Returns:
            request_id: Request identifier
        """
        event_id = str(uuid.uuid4())
        request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        event = PredictionRequestEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            vessel_id=vessel_id,
            request_id=request_id,
            data={
                "vessel_snapshot": vessel_snapshot,
                "terminal_filter": terminal_filter,
                "priority": priority
            },
            metadata={
                "triggered_by": "api",
                "correlation_id": correlation_id
            }
        ).model_dump(mode='json')

        self._publish(
            topic=settings.KAFKA_PREDICTION_REQUESTS_TOPIC,
            key=vessel_id,
            event=event
        )

        logger.info(f"Published prediction request for vessel {vessel_id}: {request_id}")
        return request_id

    def publish_prediction_result(
        self,
        vessel_id: str,
        request_id: str,
        prediction_data: dict,
        processing_time_ms: int,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Publish prediction result event.

        Args:
            vessel_id: Vessel identifier
            request_id: Original request ID
            prediction_data: Complete prediction result data
            processing_time_ms: Time taken to process prediction
            correlation_id: Optional correlation ID

        Returns:
            event_id: Generated event ID
        """
        event_id = str(uuid.uuid4())

        event = PredictionResultEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            vessel_id=vessel_id,
            request_id=request_id,
            data=prediction_data,
            metadata={
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "model_version": settings.OPENAI_MODEL
            }
        ).model_dump(mode='json')

        self._publish(
            topic=settings.KAFKA_PREDICTION_RESULTS_TOPIC,
            key=vessel_id,
            event=event
        )

        logger.info(f"Published prediction result for vessel {vessel_id}: {request_id}")
        return event_id

    def flush(self, timeout: float = 10.0):
        """
        Flush all pending messages.

        Args:
            timeout: Maximum time to wait in seconds
        """
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages failed to flush")
        else:
            logger.debug("All messages flushed successfully")

    def close(self):
        """Close producer and flush pending messages"""
        self.flush()
        logger.info("Kafka producer closed")


# Global producer instance
_producer: Optional[VesselEventProducer] = None


def get_producer() -> VesselEventProducer:
    """Get or create global producer instance"""
    global _producer
    if _producer is None:
        _producer = VesselEventProducer()
    return _producer


def close_producer():
    """Close global producer instance"""
    global _producer
    if _producer:
        _producer.close()
        _producer = None
```

---

## 7.5. Kafka Consumer Base Class

### Create `service/src/services/kafka_consumer.py`:

```python
from confluent_kafka import Consumer, KafkaError, KafkaException
import json
import logging
from typing import Callable, Optional, List
from abc import ABC, abstractmethod
import signal
import sys

from src.config import settings

logger = logging.getLogger(__name__)


class BaseKafkaConsumer(ABC):
    """
    Base class for Kafka consumers.
    Provides connection management, error handling, and graceful shutdown.
    """

    def __init__(
        self,
        topics: List[str],
        group_id: str,
        auto_offset_reset: str = "latest"
    ):
        self.topics = topics
        self.group_id = group_id
        self.running = False

        self.consumer = Consumer({
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': group_id,
            'auto.offset.reset': auto_offset_reset,
            'enable.auto.commit': False,  # Manual commit for at-least-once
            'max.poll.interval.ms': 300000,  # 5 minutes
        })

        logger.info(f"Consumer initialized: group={group_id}, topics={topics}")

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down consumer...")
        self.stop()
        sys.exit(0)

    @abstractmethod
    async def process_message(self, event: dict) -> bool:
        """
        Process a single message. Must be implemented by subclass.

        Args:
            event: Deserialized event dictionary

        Returns:
            bool: True if processing successful, False to retry
        """
        pass

    def start(self):
        """Start consuming messages"""
        self.running = True
        self.consumer.subscribe(self.topics)

        logger.info(f"Consumer started: {self.group_id}")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition - not an error
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue

                # Process message
                try:
                    event = json.loads(msg.value().decode('utf-8'))
                    logger.debug(f"Processing event: {event.get('event_id')}")

                    # Call subclass implementation
                    success = await self.process_message(event)

                    if success:
                        # Commit offset after successful processing
                        self.consumer.commit(message=msg)
                    else:
                        # Don't commit - message will be retried
                        logger.warning(f"Processing failed, will retry: {event.get('event_id')}")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in message: {e}")
                    # Commit to skip invalid message
                    self.consumer.commit(message=msg)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    # Don't commit - will retry

        except KafkaException as e:
            logger.error(f"Kafka exception: {e}")
        finally:
            self.close()

    def stop(self):
        """Stop consuming messages"""
        self.running = False
        logger.info(f"Consumer stopping: {self.group_id}")

    def close(self):
        """Close consumer connection"""
        self.consumer.close()
        logger.info(f"Consumer closed: {self.group_id}")
```

---

## 7.6. Prediction Worker Consumer

### Create `service/src/services/consumers/prediction_worker.py`:

```python
import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.services.kafka_consumer import BaseKafkaConsumer
from src.services.kafka_producer import get_producer
from src.services.prediction_engine import PredictionEngine
from src.services.rag_service import RAGService
from src.services.ai_service import AIService
from src.database.connection import AsyncSessionLocal
from src.config import settings

logger = logging.getLogger(__name__)


class PredictionWorker(BaseKafkaConsumer):
    """
    Consumer that processes prediction requests.
    Orchestrates full prediction pipeline: traditional → RAG → AI
    """

    def __init__(self):
        super().__init__(
            topics=[settings.KAFKA_PREDICTION_REQUESTS_TOPIC],
            group_id='prediction-workers',
            auto_offset_reset='latest'
        )
        self.producer = get_producer()

    async def process_message(self, event: dict) -> bool:
        """
        Process prediction request event.

        Pipeline:
        1. Extract vessel and request info
        2. Run traditional prediction engine
        3. Run RAG similarity search
        4. Run AI analysis with GPT-4o
        5. Publish prediction results

        Returns:
            bool: True if successful, False to retry
        """
        start_time = datetime.utcnow()

        try:
            vessel_id = event['vessel_id']
            request_id = event['request_id']
            vessel_snapshot = event['data']['vessel_snapshot']
            correlation_id = event['metadata'].get('correlation_id')

            logger.info(f"Processing prediction request: {request_id} for vessel {vessel_id}")

            # Create database session
            async with AsyncSessionLocal() as session:
                # Get vessel from DB (create if needed)
                from src.database.models import Vessel
                from sqlalchemy import select

                result = await session.execute(
                    select(Vessel).where(Vessel.id == vessel_id)
                )
                vessel = result.scalar_one_or_none()

                if not vessel:
                    logger.warning(f"Vessel {vessel_id} not found, creating from snapshot")
                    vessel = Vessel(
                        id=vessel_id,
                        name=f"Vessel {vessel_id}",
                        current_lat=vessel_snapshot['lat'],
                        current_lon=vessel_snapshot['lon'],
                        speed=vessel_snapshot['speed'],
                        heading=vessel_snapshot['heading']
                    )
                    session.add(vessel)
                    await session.commit()
                    await session.refresh(vessel)

                # Initialize services
                prediction_engine = PredictionEngine(session)
                rag_service = RAGService(session)
                ai_service = AIService(session)

                # Run traditional prediction engine
                predictions = await prediction_engine.analyze_vessel(vessel_id)

                if not predictions:
                    logger.info(f"No predictions found for vessel {vessel_id}")
                    return True

                # Process top prediction with AI enhancement
                for pred in predictions[:3]:  # Process top 3
                    # Get RAG similar journeys
                    from sqlalchemy import select
                    from src.database.models import Terminal

                    terminal_result = await session.execute(
                        select(Terminal).where(Terminal.id == pred.terminal_id)
                    )
                    terminal = terminal_result.scalar_one()

                    similar_journeys = await rag_service.find_similar_journeys(
                        vessel, terminal, limit=5
                    )

                    # Get AI analysis
                    ai_result = await ai_service.analyze_prediction(
                        vessel=vessel,
                        terminal=terminal,
                        base_confidence=pred.confidence_score,
                        proximity_score=pred.proximity_score,
                        speed_score=pred.speed_score,
                        heading_score=pred.heading_score,
                        similar_journeys=similar_journeys
                    )

                    # Calculate final confidence
                    rag_boost = rag_service.calculate_historical_similarity_score(similar_journeys)
                    final_confidence = pred.confidence_score + rag_boost + ai_result['adjustment']
                    final_confidence = max(0.0, min(1.0, final_confidence))

                    # Save prediction to DB
                    from src.database.models import Prediction
                    import uuid

                    db_prediction = Prediction(
                        id=str(uuid.uuid4()),
                        vessel_id=vessel_id,
                        terminal_id=pred.terminal_id,
                        confidence_score=final_confidence,
                        proximity_score=pred.proximity_score,
                        speed_score=pred.speed_score,
                        heading_score=pred.heading_score,
                        rag_score=rag_boost,
                        ai_adjustment=ai_result['adjustment'],
                        ai_reasoning=ai_result['reasoning'],
                        distance_to_terminal_km=pred.distance_km,
                        eta_hours=pred.eta_hours,
                        vessel_speed=vessel_snapshot['speed'],
                        vessel_heading=vessel_snapshot['heading'],
                        vessel_lat=vessel_snapshot['lat'],
                        vessel_lon=vessel_snapshot['lon'],
                        status='active'
                    )
                    session.add(db_prediction)
                    await session.commit()
                    await session.refresh(db_prediction)

                    # Publish prediction result event
                    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                    self.producer.publish_prediction_result(
                        vessel_id=vessel_id,
                        request_id=request_id,
                        prediction_data={
                            "prediction_id": db_prediction.id,
                            "terminal_id": pred.terminal_id,
                            "terminal_name": terminal.name,
                            "confidence_score": final_confidence,
                            "scores": {
                                "proximity": pred.proximity_score,
                                "speed": pred.speed_score,
                                "heading": pred.heading_score,
                                "rag_boost": rag_boost,
                                "ai_adjustment": ai_result['adjustment']
                            },
                            "distance_km": pred.distance_km,
                            "eta_hours": pred.eta_hours,
                            "ai_reasoning": ai_result['reasoning'],
                            "vessel_snapshot": vessel_snapshot
                        },
                        processing_time_ms=processing_time,
                        correlation_id=correlation_id
                    )

                    logger.info(
                        f"Prediction complete: {vessel_id} → {terminal.name} "
                        f"(confidence: {final_confidence:.2f})"
                    )

            return True

        except Exception as e:
            logger.error(f"Error processing prediction request: {e}", exc_info=True)
            return False  # Retry


def main():
    """Main entry point for prediction worker"""
    worker = PredictionWorker()
    worker.start()


if __name__ == "__main__":
    main()
```

---

## 7.7. Database Writer Consumer

### Create `service/src/services/consumers/prediction_persister.py`:

```python
import asyncio
import logging
from src.services.kafka_consumer import BaseKafkaConsumer
from src.database.connection import AsyncSessionLocal
from src.database.models import Prediction
from src.config import settings
from sqlalchemy import select

logger = logging.getLogger(__name__)


class PredictionPersister(BaseKafkaConsumer):
    """
    Consumer that persists prediction results to PostgreSQL.
    Part of fan-out pattern from prediction-results topic.
    """

    def __init__(self):
        super().__init__(
            topics=[settings.KAFKA_PREDICTION_RESULTS_TOPIC],
            group_id='prediction-db-writers',
            auto_offset_reset='latest'
        )

    async def process_message(self, event: dict) -> bool:
        """
        Save prediction result to database (idempotent).

        Returns:
            bool: True if successful, False to retry
        """
        try:
            prediction_id = event['data']['prediction_id']

            async with AsyncSessionLocal() as session:
                # Check if already exists (idempotency)
                result = await session.execute(
                    select(Prediction).where(Prediction.id == prediction_id)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    logger.debug(f"Prediction {prediction_id} already exists, skipping")
                    return True

                # Prediction was already saved by worker - this consumer
                # could be used for additional persistence tasks like:
                # - Archive to data warehouse
                # - Update analytics tables
                # - Cache invalidation
                # etc.

                logger.info(f"Processed prediction result: {prediction_id}")

            return True

        except Exception as e:
            logger.error(f"Error persisting prediction: {e}", exc_info=True)
            return False


def main():
    """Main entry point for prediction persister"""
    persister = PredictionPersister()
    persister.start()


if __name__ == "__main__":
    main()
```

---

## 7.8. WebSocket Broadcaster Consumer

### Create `service/src/services/consumers/websocket_broadcaster.py`:

```python
import asyncio
import logging
from src.services.kafka_consumer import BaseKafkaConsumer
from src.config import settings

logger = logging.getLogger(__name__)


class WebSocketBroadcaster(BaseKafkaConsumer):
    """
    Consumer that broadcasts prediction results to WebSocket clients.
    Part of fan-out pattern from prediction-results topic.
    """

    def __init__(self, websocket_manager):
        super().__init__(
            topics=[settings.KAFKA_PREDICTION_RESULTS_TOPIC],
            group_id='prediction-ws-broadcasters',
            auto_offset_reset='latest'
        )
        self.ws_manager = websocket_manager

    async def process_message(self, event: dict) -> bool:
        """
        Broadcast prediction result to WebSocket clients.

        Returns:
            bool: True if successful, False to retry
        """
        try:
            vessel_id = event['vessel_id']
            prediction_data = event['data']

            # Format for frontend
            ws_message = {
                "type": "prediction_created",
                "vessel_id": vessel_id,
                "prediction": {
                    "id": prediction_data['prediction_id'],
                    "terminal": {
                        "id": prediction_data['terminal_id'],
                        "name": prediction_data['terminal_name']
                    },
                    "confidence": prediction_data['confidence_score'],
                    "distance_km": prediction_data['distance_km'],
                    "eta_hours": prediction_data['eta_hours'],
                    "ai_reasoning": prediction_data['ai_reasoning'],
                    "scores": prediction_data['scores']
                }
            }

            # Broadcast to all connected clients
            await self.ws_manager.broadcast(ws_message)

            logger.info(f"Broadcasted prediction to WebSocket clients: {vessel_id}")
            return True

        except Exception as e:
            logger.error(f"Error broadcasting to WebSocket: {e}", exc_info=True)
            return False


def main():
    """Main entry point for WebSocket broadcaster"""
    # This would typically be started as part of main application
    # where WebSocketManager is available
    pass


if __name__ == "__main__":
    main()
```

---

## 7.9. Update API Endpoints

### Modify `service/src/api/routers/vessels.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from src.api.deps import get_db
from src.schemas.vessel import VesselCreate, VesselUpdate, VesselResponse
from src.database.models import Vessel
from src.services.kafka_producer import get_producer
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.put("/{vessel_id}", response_model=VesselResponse, status_code=status.HTTP_202_ACCEPTED)
async def update_vessel_position(
    vessel_id: str,
    vessel_update: VesselUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update vessel position - publishes event to Kafka for async processing.

    Returns 202 Accepted immediately - prediction processing happens asynchronously.
    """
    # Get vessel from database
    result = await db.execute(select(Vessel).where(Vessel.id == vessel_id))
    vessel = result.scalar_one_or_none()

    if not vessel:
        raise HTTPException(status_code=404, detail="Vessel not found")

    # Update vessel in database
    vessel.current_lat = vessel_update.current_lat
    vessel.current_lon = vessel_update.current_lon
    vessel.speed = vessel_update.speed
    vessel.heading = vessel_update.heading

    await db.commit()
    await db.refresh(vessel)

    # Publish position update event to Kafka
    producer = get_producer()
    event_id = producer.publish_vessel_position_update(
        vessel_id=vessel_id,
        lat=vessel.current_lat,
        lon=vessel.current_lon,
        speed=vessel.speed,
        heading=vessel.heading,
        status="underway"
    )

    logger.info(f"Position update published: {vessel_id} (event: {event_id})")

    # Return immediately - processing happens asynchronously
    return VesselResponse.from_orm(vessel)
```

### Modify `service/src/api/routers/predictions.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.services.kafka_producer import get_producer
from src.database.models import Vessel
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze/{vessel_id}")
async def trigger_prediction_analysis(
    vessel_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger async prediction analysis for vessel.

    Returns request_id for tracking - actual prediction happens asynchronously.
    """
    # Get vessel
    result = await db.execute(select(Vessel).where(Vessel.id == vessel_id))
    vessel = result.scalar_one_or_none()

    if not vessel:
        raise HTTPException(status_code=404, detail="Vessel not found")

    # Publish prediction request event
    producer = get_producer()
    request_id = producer.publish_prediction_request(
        vessel_id=vessel_id,
        vessel_snapshot={
            "lat": vessel.current_lat,
            "lon": vessel.current_lon,
            "speed": vessel.speed,
            "heading": vessel.heading,
            "status": "underway"
        },
        priority="normal"
    )

    logger.info(f"Prediction analysis requested: {vessel_id} (request: {request_id})")

    return {
        "status": "accepted",
        "request_id": request_id,
        "message": "Prediction analysis queued for processing"
    }
```

---

## 7.10. Start Consumers as Background Services

### Create `service/src/workers.py`:

```python
import asyncio
import logging
from multiprocessing import Process

from src.services.consumers.prediction_worker import PredictionWorker
from src.services.consumers.prediction_persister import PredictionPersister
from src.services.consumers.websocket_broadcaster import WebSocketBroadcaster

logger = logging.getLogger(__name__)


def start_prediction_worker():
    """Start prediction worker consumer"""
    worker = PredictionWorker()
    worker.start()


def start_prediction_persister():
    """Start prediction persister consumer"""
    persister = PredictionPersister()
    persister.start()


def start_websocket_broadcaster(websocket_manager):
    """Start WebSocket broadcaster consumer"""
    broadcaster = WebSocketBroadcaster(websocket_manager)
    broadcaster.start()


def start_all_consumers(websocket_manager=None):
    """Start all consumer processes"""
    processes = []

    # Start prediction worker
    p1 = Process(target=start_prediction_worker)
    p1.start()
    processes.append(p1)
    logger.info("Started prediction worker process")

    # Start prediction persister
    p2 = Process(target=start_prediction_persister)
    p2.start()
    processes.append(p2)
    logger.info("Started prediction persister process")

    # WebSocket broadcaster runs in main process
    # (needs access to WebSocketManager)

    return processes
```

### Update `service/src/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.config import settings
from src.api.routers import vessels, predictions, terminals, admin, websocket
from src.services.kafka_producer import get_producer, close_producer
from src.workers import start_all_consumers

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(vessels.router, prefix=f"{settings.API_V1_PREFIX}/vessels", tags=["vessels"])
app.include_router(predictions.router, prefix=f"{settings.API_V1_PREFIX}/predictions", tags=["predictions"])
app.include_router(terminals.router, prefix=f"{settings.API_V1_PREFIX}/terminals", tags=["terminals"])
app.include_router(admin.router, prefix=f"{settings.API_V1_PREFIX}/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.on_event("startup")
async def startup_event():
    """Initialize Kafka producer and start consumers on startup"""
    logger.info("Starting Vessel Track API...")

    # Initialize Kafka producer
    get_producer()
    logger.info("Kafka producer initialized")

    # Start consumer processes
    # Note: In production, run consumers as separate services
    # For development, can start as background processes
    # consumer_processes = start_all_consumers()
    # app.state.consumer_processes = consumer_processes

    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Close Kafka connections on shutdown"""
    logger.info("Shutting down Vessel Track API...")

    # Close Kafka producer
    close_producer()
    logger.info("Kafka producer closed")

    # Stop consumer processes
    # if hasattr(app.state, 'consumer_processes'):
    #     for process in app.state.consumer_processes:
    #         process.terminate()
    #         process.join()
    #     logger.info("Consumer processes stopped")

    logger.info("Shutdown complete")


@app.get("/")
async def root():
    return {
        "name": "Vessel Track API",
        "version": "1.0.0",
        "status": "running",
        "architecture": "event-driven (Kafka)"
    }
```

---

## Verification Checklist

- [ ] Docker Compose starts all services (PostgreSQL, Redis, Zookeeper, Kafka)
- [ ] All Kafka health checks pass
- [ ] Kafka topics created with correct configuration
- [ ] Python dependencies installed (confluent-kafka)
- [ ] Configuration updated with Kafka settings
- [ ] Event schemas defined and validated
- [ ] Kafka producer service implemented
- [ ] Kafka consumer base class implemented
- [ ] Prediction worker consumer implemented
- [ ] API endpoints updated to publish events
- [ ] API returns 202 Accepted immediately (<50ms)
- [ ] Prediction worker processes requests asynchronously
- [ ] Prediction results published to results topic
- [ ] Multiple consumers receive prediction results
- [ ] WebSocket clients receive real-time updates
- [ ] End-to-end latency: event published → result received < 5 seconds

---

## Testing

### Manual Testing

```bash
# 1. Start all services
cd service
docker-compose up -d

# 2. Verify Kafka is running
docker-compose ps
docker-compose logs kafka

# 3. Create topics
./scripts/create_kafka_topics.sh

# 4. Start API server
poetry run uvicorn src.main:app --reload

# 5. In separate terminal, start prediction worker
poetry run python src/services/consumers/prediction_worker.py

# 6. Test vessel position update (should publish event)
curl -X PUT http://localhost:8000/api/v1/vessels/lng-001 \
  -H "Content-Type: application/json" \
  -d '{
    "current_lat": 29.7604,
    "current_lon": -95.3698,
    "speed": 12.5,
    "heading": 275.0
  }'

# 7. Check Kafka messages
docker exec vessel_track_kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic vessel-position-updates \
  --from-beginning

# 8. Check prediction results
docker exec vessel_track_kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic prediction-results \
  --from-beginning
```

### Consumer Group Monitoring

```bash
# Check consumer group lag
docker exec vessel_track_kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group prediction-workers \
  --describe

# Should show:
# - Current offset
# - Log end offset
# - Lag (should be close to 0)
```

---

## Troubleshooting

### Kafka Won't Start

**Symptom**: `vessel_track_kafka` container unhealthy

**Solution**:
```bash
# Check logs
docker-compose logs kafka

# Common issues:
# 1. Zookeeper not ready - wait for zookeeper health check
# 2. Port 9092 already in use - check with: lsof -i :9092

# Restart services
docker-compose down
docker-compose up -d
```

### Messages Not Being Consumed

**Symptom**: Consumer lag increasing

**Solution**:
```bash
# 1. Check consumer is running
ps aux | grep prediction_worker

# 2. Check consumer logs
tail -f logs/prediction_worker.log

# 3. Check consumer group status
docker exec vessel_track_kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group prediction-workers \
  --describe

# 4. Check for errors in consumer code
# 5. Increase consumer parallelism (start more instances)
```

### High Latency

**Symptom**: End-to-end latency > 10 seconds

**Solution**:
1. Check GPT-4o API latency (should be 2-5s)
2. Add more prediction worker instances
3. Optimize database queries
4. Check network between containers

---

## Production Considerations

### Separate Consumer Services

In production, run consumers as separate services (not in-process):

```yaml
# docker-compose.prod.yml
  prediction-worker:
    build: .
    command: python src/services/consumers/prediction_worker.py
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
    depends_on:
      - kafka
      - postgres
    deploy:
      replicas: 3  # Scale horizontally
```

### Monitoring

Add monitoring for:
- Consumer lag (alert if > 1000 messages)
- Processing time (alert if > 10 seconds)
- Error rate (alert if > 5%)
- Throughput (messages/second)

### Replication

In production, use replication factor ≥ 2:
```bash
--replication-factor 2
```

### Authentication

Enable SASL authentication:
```yaml
KAFKA_SASL_ENABLED_MECHANISMS: PLAIN
KAFKA_SASL_MECHANISM_INTER_BROKER_PROTOCOL: PLAIN
```

---

## Next Steps

After completing Phase 7:

1. **Monitor Performance**: Watch consumer lag and latency metrics
2. **Scale Workers**: Add more prediction worker instances based on load
3. **Add More Streams**: Consider adding streams for:
   - Vessel arrivals (learning feedback)
   - Behavior events (speed/course changes)
   - Notifications (Slack, email)
4. **Optimize**: Tune Kafka parameters based on throughput needs
5. **Production Deploy**: Move to separate consumer services with proper monitoring

---

**Status**: ✅ Phase 7 Complete - Event-Driven Architecture with Kafka
