# Kafka Event-Driven Architecture

## Overview

Vessel Track uses Apache Kafka as the core event streaming platform to enable asynchronous, scalable, and resilient processing of vessel tracking and prediction workflows. This document defines all event streams, message schemas, and processing patterns.

## Event-Driven Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Map Display │  │  WebSocket   │  │  API Client  │              │
│  │   (Mapbox)   │  │   Client     │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└────────────────┬────────────────┬───────────────────────────────────┘
                 │ HTTP           │ WebSocket
                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  REST API    │  │  WebSocket   │  │  Background  │              │
│  │  Endpoints   │  │   Manager    │  │    Tasks     │              │
│  └──────┬───────┘  └──────▲───────┘  └──────────────┘              │
│         │                  │                                         │
│         │ publish          │ subscribe                               │
│         ▼                  │                                         │
│  ┌─────────────────────────────────────────────────┐                │
│  │           Kafka Event Broker                    │                │
│  │  ┌──────────────────┐  ┌──────────────────┐    │                │
│  │  │ vessel-position- │  │ prediction-      │    │                │
│  │  │    updates       │  │   requests       │    │                │
│  │  └──────────────────┘  └──────────────────┘    │                │
│  │  ┌──────────────────┐                          │                │
│  │  │ prediction-      │                          │                │
│  │  │   results        │                          │                │
│  │  └──────────────────┘                          │                │
│  └─────────────────────────────────────────────────┘                │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────┐    ┌─────────────┐   ┌─────────────┐             │
│  │  Position   │    │ Prediction  │   │  Result     │             │
│  │  Consumer   │    │   Worker    │   │ Consumers   │             │
│  │             │    │             │   │ (DB/WS/     │             │
│  │             │    │             │   │  Slack)     │             │
│  └─────────────┘    └─────────────┘   └─────────────┘             │
└────────────────┬────────────────┬────────────────┬──────────────────┘
                 │                │                │
                 ▼                ▼                ▼
┌──────────────────────┐  ┌────────────────────────┐
│  PostgreSQL          │  │  Redis                 │
│  + pgvector          │  │  (Caching)             │
│  (Vector DB + Data)  │  │                        │
└──────────────────────┘  └────────────────────────┘
```

## Benefits of Event-Driven Architecture

### 1. Asynchronous Processing
- **API Response Time**: 5-10 seconds → <50ms
- **Non-blocking**: Clients don't wait for slow operations (AI analysis, embeddings)
- **Improved UX**: Instant feedback with progressive updates

### 2. Horizontal Scalability
- **Multiple Workers**: Scale prediction processing independently
- **Partition Distribution**: Kafka distributes load across consumers
- **Elastic Scaling**: Add/remove consumers without code changes

### 3. Decoupling & Resilience
- **Service Independence**: Services communicate via events, not direct calls
- **Fault Tolerance**: Failed consumers can retry without affecting producers
- **Evolution**: Change consumer logic without touching producers

### 4. Observability & Audit Trail
- **Complete Event Log**: Every action recorded in Kafka
- **Replay Capability**: Reprocess historical events for debugging or retraining
- **Traceability**: Correlation IDs track requests through entire pipeline

### 5. Multiple Consumers Pattern
- **Fan-out**: Single prediction result consumed by DB writer, WebSocket, Slack, learning service
- **Specialized Processing**: Each consumer handles one responsibility
- **Independent Scaling**: Scale each consumer based on its needs

## Event Stream Catalog

### 1. vessel-position-updates

**Purpose**: Stream of real-time vessel location and state changes

**Producers**:
- **Vessel Update API** (`POST /api/v1/vessels/{id}`) - Manual position updates
- **AIS Data Ingestion Service** (future) - Automatic position updates from AIS feeds
- **Test Data Generators** - Simulated vessel movements for testing

**Consumers**:
- **Prediction Request Trigger** - Checks if vessel is within range of terminals and publishes prediction requests
- **WebSocket Broadcaster** - Pushes real-time position updates to connected frontend clients
- **Journey Tracker** - Records historical vessel paths for journey analysis
- **Behavior Event Detector** (future) - Identifies significant speed/course changes

**Message Schema**:
```json
{
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
```

**Schema Definition** (Pydantic):
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

class VesselPositionData(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    speed: float = Field(..., ge=0)
    heading: float = Field(..., ge=0, lt=360)
    status: str

class VesselPositionUpdateEvent(BaseModel):
    event_id: str
    event_type: Literal["vessel.position.updated"]
    timestamp: datetime
    vessel_id: str
    data: VesselPositionData
    metadata: dict
```

**Topic Configuration**:
- **Topic Name**: `vessel-position-updates`
- **Partitions**: 3
- **Partition Key**: `vessel_id` (ensures ordering per vessel)
- **Retention**: 7 days
- **Replication Factor**: 1 (dev), 2+ (production)
- **Cleanup Policy**: delete (time-based retention)

**Processing Characteristics**:
- **Throughput**: 100-1000 messages/second
- **Latency**: <50ms producer to consumer
- **Ordering**: Guaranteed per vessel_id (same partition)

---

### 2. prediction-requests

**Purpose**: Trigger asynchronous prediction analysis for vessel-terminal pairs

**Producers**:
- **Position Consumer** - When vessel enters terminal proximity range (<500km)
- **Scheduled Refresh Jobs** - Periodic re-analysis of active vessels
- **Manual Prediction API** (`POST /api/v1/predictions/analyze`) - On-demand analysis

**Consumers**:
- **Prediction Pipeline Workers** - Consume requests and orchestrate full prediction pipeline (traditional algorithm + RAG + AI analysis)

**Message Schema**:
```json
{
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
    "terminal_filter": null,
    "priority": "normal"
  },
  "metadata": {
    "triggered_by": "position_update",
    "correlation_id": "corr-abc123",
    "client_id": "api_user_xyz"
  }
}
```

**Schema Definition** (Pydantic):
```python
class VesselSnapshot(BaseModel):
    lat: float
    lon: float
    speed: float
    heading: float
    status: str

class PredictionRequestData(BaseModel):
    vessel_snapshot: VesselSnapshot
    terminal_filter: Optional[int] = None  # Analyze specific terminal only
    priority: Literal["low", "normal", "high"] = "normal"

class PredictionRequestEvent(BaseModel):
    event_id: str
    event_type: Literal["prediction.analysis.requested"]
    timestamp: datetime
    vessel_id: str
    request_id: str
    data: PredictionRequestData
    metadata: dict
```

**Topic Configuration**:
- **Topic Name**: `prediction-requests`
- **Partitions**: 5
- **Partition Key**: `vessel_id` (distributes load, maintains per-vessel ordering)
- **Retention**: 24 hours
- **Replication Factor**: 1 (dev), 2+ (production)
- **Cleanup Policy**: delete

**Processing Characteristics**:
- **Throughput**: 10-100 messages/second
- **Latency**: <100ms producer to consumer
- **Processing Time**: 2-5 seconds (includes GPT-4o API call)
- **Parallelism**: Multiple workers process different vessels simultaneously

**Consumer Behavior**:
- **Consumer Group**: `prediction-workers`
- **Parallelism**: 3-10 instances (scale based on load)
- **Idempotency**: Check if recent prediction exists before processing
- **Error Handling**: Retry up to 3 times, then send to dead letter queue

---

### 3. prediction-results

**Purpose**: Completed predictions ready for storage, notification, and display

**Producers**:
- **Prediction Pipeline Workers** - After completing full prediction analysis

**Consumers**:
- **Database Writer** - Persists predictions to PostgreSQL
- **WebSocket Broadcaster** - Pushes prediction to connected frontend clients
- **Slack Notifier** - Sends alerts for high-confidence predictions (≥80%)
- **Learning Service** - Prepares prediction for outcome tracking and feedback loop

**Message Schema**:
```json
{
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
    "ai_reasoning": "Vessel is maintaining optimal approach speed of 12.5 knots and heading directly toward Sabine Pass LNG. Historical data shows similar successful approaches from this position. High confidence this is the intended destination.",
    "vessel_snapshot": {
      "lat": 29.7604,
      "lon": -95.3698,
      "speed": 12.5,
      "heading": 275.0
    }
  },
  "metadata": {
    "processing_time_ms": 2340,
    "correlation_id": "corr-abc123",
    "model_version": "gpt-4o"
  }
}
```

**Schema Definition** (Pydantic):
```python
class PredictionScores(BaseModel):
    proximity: float
    speed: float
    heading: float
    rag_boost: float
    ai_adjustment: float

class PredictionResultData(BaseModel):
    prediction_id: str
    terminal_id: int
    terminal_name: str
    confidence_score: float = Field(..., ge=0, le=1)
    scores: PredictionScores
    distance_km: float
    eta_hours: Optional[float]
    ai_reasoning: str
    vessel_snapshot: VesselSnapshot

class PredictionResultEvent(BaseModel):
    event_id: str
    event_type: Literal["prediction.analysis.completed"]
    timestamp: datetime
    vessel_id: str
    request_id: str
    data: PredictionResultData
    metadata: dict
```

**Topic Configuration**:
- **Topic Name**: `prediction-results`
- **Partitions**: 3
- **Partition Key**: `vessel_id`
- **Retention**: 7 days
- **Replication Factor**: 1 (dev), 2+ (production)
- **Cleanup Policy**: delete

**Processing Characteristics**:
- **Throughput**: 10-100 messages/second
- **Latency**: <50ms producer to all consumers
- **Fan-out**: 4 independent consumers process each message
- **Idempotency**: Each consumer handles duplicates gracefully

**Consumer Groups**:
- `prediction-db-writers` - Database persistence
- `prediction-ws-broadcasters` - WebSocket notifications
- `prediction-slack-notifiers` - Slack alerts
- `prediction-learning-processors` - Learning service

---

## Event Processing Patterns

### Producer Patterns

#### Fire-and-Forget
```python
from confluent_kafka import Producer

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def publish_position_update(vessel_id: str, lat: float, lon: float, speed: float, heading: float):
    event = {
        'event_id': str(uuid.uuid4()),
        'event_type': 'vessel.position.updated',
        'timestamp': datetime.utcnow().isoformat(),
        'vessel_id': vessel_id,
        'data': {'lat': lat, 'lon': lon, 'speed': speed, 'heading': heading}
    }

    producer.produce(
        topic='vessel-position-updates',
        key=vessel_id.encode('utf-8'),
        value=json.dumps(event).encode('utf-8')
    )
    producer.poll(0)  # Trigger callbacks
```

#### With Delivery Callback
```python
def delivery_callback(err, msg):
    if err:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

producer.produce(
    topic='prediction-requests',
    key=vessel_id.encode('utf-8'),
    value=json.dumps(event).encode('utf-8'),
    callback=delivery_callback
)
producer.flush()  # Wait for delivery
```

### Consumer Patterns

#### At-Least-Once Processing
```python
from confluent_kafka import Consumer, KafkaError

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'prediction-workers',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False  # Manual commit for at-least-once
})

consumer.subscribe(['prediction-requests'])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            continue
        else:
            logger.error(f"Consumer error: {msg.error()}")
            continue

    try:
        # Process message
        event = json.loads(msg.value().decode('utf-8'))
        process_prediction_request(event)

        # Commit offset only after successful processing
        consumer.commit(message=msg)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        # Don't commit - message will be reprocessed
```

#### Idempotent Processing
```python
async def process_prediction_result(event: dict):
    prediction_id = event['data']['prediction_id']

    # Check if already processed
    existing = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    if existing.scalar_one_or_none():
        logger.info(f"Prediction {prediction_id} already exists, skipping")
        return

    # Process and store
    await save_prediction_to_db(event)
```

### Error Handling

#### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def process_with_retry(event: dict):
    # Processing logic
    result = await call_external_service(event)
    return result
```

#### Dead Letter Queue
```python
def process_message(msg):
    try:
        event = json.loads(msg.value().decode('utf-8'))
        process_prediction_request(event)
        consumer.commit(message=msg)
    except Exception as e:
        logger.error(f"Failed to process message: {e}")

        # Send to DLQ after max retries
        if get_retry_count(msg) >= 3:
            send_to_dlq(msg, error=str(e))
            consumer.commit(message=msg)  # Commit to move forward
        else:
            # Don't commit - will be retried
            pass
```

## Performance Characteristics

### Throughput
- **Vessel Position Updates**: 100-1,000 msg/sec
- **Prediction Requests**: 10-100 msg/sec
- **Prediction Results**: 10-100 msg/sec

### Latency
- **Producer to Kafka**: <10ms
- **Kafka to Consumer**: <50ms
- **End-to-End** (position update → prediction result): 2-5 seconds

### Scaling
- **Horizontal**: Add consumer instances to consumer group
- **Vertical**: Increase Kafka broker resources
- **Partitioning**: More partitions = more parallelism

### Resource Requirements

**Development**:
- Kafka: 512MB RAM, 1 CPU
- Zookeeper: 256MB RAM, 1 CPU

**Production** (100 vessels, 1000 predictions/hour):
- Kafka: 2GB RAM, 2 CPU
- Zookeeper: 512MB RAM, 1 CPU
- Storage: 10GB for 7-day retention

## Monitoring & Observability

### Key Metrics

**Producer Metrics**:
- `kafka.producer.messages.sent` - Total messages published
- `kafka.producer.errors` - Failed publishes
- `kafka.producer.latency` - Time to publish

**Consumer Metrics**:
- `kafka.consumer.messages.consumed` - Total messages processed
- `kafka.consumer.lag` - Messages behind head of partition
- `kafka.consumer.processing_time` - Time to process message

**Topic Metrics**:
- `kafka.topic.size` - Total bytes stored
- `kafka.topic.messages` - Total message count
- `kafka.topic.partition_lag` - Per-partition lag

### Health Checks

```python
from confluent_kafka.admin import AdminClient

def check_kafka_health():
    admin = AdminClient({'bootstrap.servers': 'localhost:9092'})

    # Check broker connectivity
    metadata = admin.list_topics(timeout=5)

    # Check topic existence
    topics = ['vessel-position-updates', 'prediction-requests', 'prediction-results']
    for topic in topics:
        if topic not in metadata.topics:
            raise Exception(f"Topic {topic} does not exist")

    return {"status": "healthy", "brokers": len(metadata.brokers)}
```

## Operational Considerations

### Topic Creation
Topics should be created during setup with proper configuration:

```bash
# Create topics with optimal configuration
kafka-topics --bootstrap-server localhost:9092 --create \
  --topic vessel-position-updates \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config compression.type=gzip

kafka-topics --bootstrap-server localhost:9092 --create \
  --topic prediction-requests \
  --partitions 5 \
  --replication-factor 1 \
  --config retention.ms=86400000

kafka-topics --bootstrap-server localhost:9092 --create \
  --topic prediction-results \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000
```

### Consumer Group Management

```bash
# List consumer groups
kafka-consumer-groups --bootstrap-server localhost:9092 --list

# Describe consumer group lag
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group prediction-workers \
  --describe

# Reset consumer group offset (re-process from beginning)
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group prediction-workers \
  --reset-offsets --to-earliest \
  --topic prediction-requests \
  --execute
```

### Troubleshooting

**High Consumer Lag**:
- Increase consumer parallelism (add more instances)
- Check for slow processing (optimize consumer logic)
- Verify consumer group is balanced across partitions

**Message Loss**:
- Check producer acknowledgment settings (`acks=all`)
- Verify replication factor ≥ 2 in production
- Review consumer commit strategy

**Duplicate Messages**:
- Implement idempotent processing in consumers
- Use unique event_id to detect duplicates
- Consider exactly-once semantics (transactional producer/consumer)

## Security Considerations

### Authentication (Production)
```yaml
# SASL/PLAIN authentication
security.protocol: SASL_PLAINTEXT
sasl.mechanism: PLAIN
sasl.username: vessel_track_user
sasl.password: <secure_password>
```

### Authorization (ACLs)
```bash
# Grant producer permissions
kafka-acls --bootstrap-server localhost:9092 \
  --add --allow-principal User:api_service \
  --operation Write --topic vessel-position-updates

# Grant consumer permissions
kafka-acls --bootstrap-server localhost:9092 \
  --add --allow-principal User:prediction_worker \
  --operation Read --topic prediction-requests \
  --group prediction-workers
```

### Encryption
- **In-transit**: TLS encryption (SSL) between clients and brokers
- **At-rest**: Disk encryption for Kafka data directories

## Migration from Synchronous Architecture

See [Phase 7: Kafka Integration](tasks/phase7-kafka-integration.md) for complete step-by-step migration guide.

**Key Migration Steps**:
1. Add Kafka infrastructure (docker-compose)
2. Implement event producers and consumers
3. Run hybrid mode (sync + async) for validation
4. Switch API endpoints to event-driven
5. Remove synchronous processing code
6. Monitor and optimize

---

**Related Documentation**:
- [Phase 7: Kafka Integration Implementation Guide](tasks/phase7-kafka-integration.md)
- [Architecture Overview](architecture.md)
- [Technology Stack](tech-stack.md)
