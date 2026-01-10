# Features

## Core Features (Implemented)

### 1. Event-Driven Architecture

**Purpose**: Asynchronous, scalable processing using Apache Kafka

**Key Benefits**:
- **Fast API Response**: <50ms (vs 5-10s synchronous)
- **Horizontal Scalability**: Add workers to increase throughput
- **Resilience**: Automatic retries, fault tolerance
- **Complete Audit Trail**: All events logged in Kafka
- **Replay Capability**: Reprocess historical events

**Event Streams**:
- `vessel-position-updates` - Real-time vessel location changes
- `prediction-requests` - Trigger async prediction analysis
- `prediction-results` - Completed predictions for multiple consumers

**Processing Pattern**:
- **Producers**: API endpoints publish events immediately
- **Consumers**: Worker pools process events in parallel
- **Fan-out**: Single prediction result consumed by DB writer, WebSocket broadcaster, Slack notifier, learning service

**Scalability**:
- 3-10 prediction workers (scale based on load)
- Kafka partitioning distributes load across consumers
- Independent scaling of each consumer type

**Technology**: Apache Kafka 3.x with Zookeeper, confluent-kafka-python client

---

### 2. Interactive Map Visualization
- **Full-screen Mapbox GL map** with dark theme
- **Vessel markers** showing real-time positions
- **Terminal markers** for LNG import/export facilities worldwide
- **Navigation controls**: Zoom, pan, compass
- **Auto-fit bounds** to display all vessels
- **Real-time updates** via WebSocket

### 2. AI-Powered Destination Prediction

#### Hybrid Prediction System
Combines three approaches for maximum accuracy:

**Traditional Algorithm (Base Scoring)**
- **Proximity Analysis**: Exponential decay scoring based on distance to terminal
- **Speed Analysis**: Identifies optimal approach speeds (8-15 knots)
- **Heading Analysis**: Calculates angular alignment with terminal bearing
- Base confidence score: weighted combination (40% proximity, 30% speed, 30% heading)

**RAG (Retrieval-Augmented Generation)**
- **Historical Pattern Matching**: Uses pgvector to find similar past journeys
- **Vector Similarity Search**: Embeddings of journey characteristics
- **Contextual Boost**: Adds up to +0.3 confidence based on historical similarity
- Learns from completed journeys to improve predictions

**GPT-4o AI Analysis**
- **Holistic Context Understanding**: Analyzes all available data
- **Confidence Adjustment**: -0.3 to +0.3 based on AI reasoning
- **Natural Language Explanation**: Human-readable justification for predictions
- **Key Factor Identification**: Highlights most important decision factors
- Considers edge cases and nuances traditional algorithms miss

#### Final Confidence Score
```
final_confidence = base_algorithm_score + rag_historical_score + ai_adjustment
```

### 3. Prediction Display

**Vessel Popups Show:**
- Vessel name, type, speed, heading, position
- **Predicted destination** with terminal name
- **Confidence percentage** with color-coded visualization
  - Green (≥80%): High confidence
  - Yellow (60-79%): Medium confidence
  - Red (<60%): Low confidence
- **Distance to terminal** in kilometers
- **Estimated Time of Arrival (ETA)** in hours
- **AI Reasoning**: Natural language explanation from GPT-4o
- **Score breakdown**: Proximity, speed, heading, historical, AI adjustment
- Multiple predictions if confidence threshold met for several terminals

### 4. Continuous Learning System

**Outcome Tracking**
- Monitors vessel arrivals at predicted terminals
- Automatic confirmation when vessel enters terminal approach zone
- Accuracy scoring for confirmed predictions

**Feedback Loop**
- **Embedding Generation**: Completed predictions stored as vectors
- **RAG Database Update**: Future predictions benefit from past outcomes
- **Approach Behavior Learning**: Tracks typical patterns per terminal/vessel
- **Behavior Event Detection**: Identifies significant speed/course changes

**Metrics Tracked**
- Prediction accuracy scores
- Average confidence levels
- Successful vs. failed predictions
- Learning curve over time

### 5. Real-time Data Updates

**WebSocket Integration**
- **Live vessel position updates** broadcast to all connected clients
- **New prediction alerts** when high-confidence predictions created
- **Subscription management**: Subscribe to specific vessels or all
- **Auto-reconnect**: Resilient connection with exponential backoff

### 6. Terminal Database

**15 Major LNG Terminals Worldwide:**

*USA Export Terminals*
- Sabine Pass LNG (Gulf of Mexico)
- Cameron LNG (Gulf of Mexico)
- Freeport LNG (Gulf of Mexico)

*Asia Import Terminals*
- Tokyo Gas Negishi (Japan)
- Incheon LNG Terminal (South Korea)
- Guangdong Dapeng LNG (China)

*Europe Import Terminals*
- Gate Terminal Rotterdam (Netherlands)
- Zeebrugge LNG Terminal (Belgium)
- Montoir-de-Bretagne (France)
- South Hook LNG (United Kingdom)

*Other Key Terminals*
- Ras Laffan LNG (Qatar) - Major export
- Gorgon LNG (Australia)
- Queensland Curtis LNG (Australia)
- Dahej LNG Terminal (India)
- Guanabara Bay LNG (Brazil)

### 7. Notifications & Alerts

**Slack Integration**
- **High-confidence alerts** (≥80%) sent to Slack channel
- Formatted messages with:
  - Vessel and terminal names
  - Confidence percentage
  - ETA and distance
  - AI reasoning
  - Score breakdown
- Color-coded based on confidence level

### 8. Admin & Metrics

**System Health**
- Database connectivity check
- Service status monitoring

**Metrics Dashboard**
- Total vessels tracked
- Active predictions count
- Confirmed predictions count
- Average prediction accuracy
- RAG embeddings stored

### 9. REST API

**Vessel Endpoints**
- `GET /api/v1/vessels` - List all vessels
- `GET /api/v1/vessels/{id}` - Get vessel with active predictions
- `POST /api/v1/vessels` - Create new vessel
- `PUT /api/v1/vessels/{id}` - Update vessel position/state

**Prediction Endpoints**
- `GET /api/v1/predictions/active` - List all active predictions
- `GET /api/v1/predictions/{id}` - Get prediction details
- `POST /api/v1/predictions/analyze` - Trigger AI prediction for vessel
- `POST /api/v1/predictions/{id}/confirm` - Confirm prediction outcome

**Terminal Endpoints**
- `GET /api/v1/terminals` - List all LNG terminals
- `GET /api/v1/terminals/{id}` - Get terminal details

**Admin Endpoints**
- `GET /api/v1/admin/health` - Health check
- `GET /api/v1/admin/metrics` - System metrics
- `GET /api/v1/admin/accuracy` - Prediction accuracy statistics

## Implementation Phases

### ✅ Phase 1: Foundation & Database (Days 1-4)
- ✅ FastAPI project structure
- ✅ Docker setup (PostgreSQL + pgvector + Redis)
- ✅ Database models (8 tables)
- ✅ Alembic migrations
- ✅ Terminal seed data

### ✅ Phase 2: Core Services (Days 5-12)
- ✅ Geographic utilities (Haversine, bearing calculations)
- ✅ Traditional prediction engine
- ✅ Embedding service (OpenAI)
- ✅ RAG service (pgvector similarity search)

### ✅ Phase 3: AI Integration (Days 13-17)
- ✅ GPT-4o AI service
- ✅ Prompt engineering for holistic analysis
- ✅ Complete prediction pipeline
- ✅ Slack notification service

### ✅ Phase 4: Learning System (Days 18-21)
- ✅ Learning service for outcome processing
- ✅ Accuracy calculation
- ✅ Approach behavior updates
- ✅ Behavior event detection
- ✅ Background task runner

### ✅ Phase 5: API & WebSocket (Days 22-26)
- ✅ FastAPI REST endpoints
- ✅ Pydantic schemas
- ✅ WebSocket manager
- ✅ Admin/metrics endpoints
- ✅ CORS configuration

### ✅ Phase 6: Frontend Integration (Days 27-28)
- ✅ API client library
- ✅ WebSocket client
- ✅ Real-time state management
- ✅ Prediction display in popups
- ✅ Terminal markers

### ✅ Phase 7: Kafka Event-Driven Architecture (Days 29-33)
- ✅ Kafka & Zookeeper infrastructure
- ✅ Event schemas (Pydantic models)
- ✅ Kafka producer service
- ✅ Kafka consumer base class
- ✅ Prediction worker consumer
- ✅ Event-driven API endpoints
- ✅ Async prediction processing
- ✅ Fan-out pattern for prediction results

## Planned Enhancements

### Short-term
- **AIS Data Integration**: Real vessel tracking from AIS providers
- **Historical Tracks**: Display vessel journey trails
- **Filtering & Search**: Find specific vessels/terminals
- **Export Functionality**: Download prediction reports

### Medium-term
- **Multi-vessel Fleet View**: Track entire fleets
- **Custom Alerts**: User-defined notification rules (easy with Kafka event consumers)
- **Analytics Dashboard**: Detailed prediction performance metrics
- **Mobile App**: Native iOS/Android applications
- **Weather Integration**: Overlay weather conditions
- **Additional Event Streams**: Vessel arrivals, behavior events, learning feedback
- **Dead Letter Queue**: Handle failed predictions systematically
- **Event Replay**: Reprocess historical data for model retraining

### Long-term
- **Route Optimization**: Suggest optimal routes
- **Fuel Efficiency**: Predict and optimize fuel consumption
- **Port Congestion Prediction**: Forecast terminal wait times
- **Multi-commodity Support**: Expand beyond LNG (oil tankers, cargo ships)
- **Collaborative Features**: Multi-user workspaces
