from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Vessel(Base):
    __tablename__ = "vessels"

    id = Column(String(50), primary_key=True)  # e.g., 'lng-001'
    name = Column(String(255), nullable=False, index=True)
    current_lat = Column(Float, nullable=False)
    current_lon = Column(Float, nullable=False)
    heading = Column(Float)  # 0-360 degrees
    speed = Column(Float)  # knots
    vessel_type = Column(String(50), default='lng_tanker')
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    mmsi = Column(String(20), unique=True, index=True, nullable=True)
    imo = Column(String(20), unique=True, index=True, nullable=True)
    status = Column(String(50))  # 'underway', 'at_anchor', 'moored'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    journeys = relationship("VesselJourney", back_populates="vessel")
    predictions = relationship("Prediction", back_populates="vessel")
    behavior_events = relationship("BehaviorEvent", back_populates="vessel")


class Terminal(Base):
    __tablename__ = "terminals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    code = Column(String(10), unique=True, index=True)  # e.g., 'SABINE'
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    country = Column(String(100))
    region = Column(String(100))
    capacity_bcm_year = Column(Float)  # Billion cubic meters per year
    terminal_type = Column(String(50))  # 'export' or 'import'
    approach_zone_radius_km = Column(Float, default=50.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    destination_journeys = relationship("VesselJourney", foreign_keys="VesselJourney.destination_terminal_id", back_populates="destination")
    predictions = relationship("Prediction", back_populates="terminal")
    approach_behaviors = relationship("TerminalApproachBehavior", back_populates="terminal")


class VesselJourney(Base):
    __tablename__ = "vessel_journeys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=False, index=True)
    origin_terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=True)
    destination_terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=False, index=True)
    departure_time = Column(DateTime(timezone=True))
    arrival_time = Column(DateTime(timezone=True))
    duration_hours = Column(Float)
    avg_speed = Column(Float)
    distance_nm = Column(Float)  # Nautical miles
    route_polyline = Column(Text)  # JSON array of [lat, lon] points
    completed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    vessel = relationship("Vessel", back_populates="journeys")
    origin = relationship("Terminal", foreign_keys=[origin_terminal_id])
    destination = relationship("Terminal", foreign_keys=[destination_terminal_id], back_populates="destination_journeys")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=False, index=True)
    terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=False, index=True)
    prediction_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    vessel_lat = Column(Float, nullable=False)
    vessel_lon = Column(Float, nullable=False)
    vessel_speed = Column(Float)
    vessel_heading = Column(Float)

    # Prediction metrics
    confidence_score = Column(Float)  # 0.0 - 1.0
    distance_to_terminal_km = Column(Float)
    eta_hours = Column(Float)
    predicted_arrival = Column(DateTime(timezone=True))

    # Score breakdown
    proximity_score = Column(Float)
    speed_score = Column(Float)
    heading_score = Column(Float)
    historical_similarity_score = Column(Float)
    ai_confidence_adjustment = Column(Float)  # -0.3 to +0.3
    ai_reasoning = Column(Text)  # GPT-4o explanation

    # Outcome tracking
    status = Column(String(50), default='active', index=True)  # active/confirmed/incorrect/expired
    actual_arrival_time = Column(DateTime(timezone=True), nullable=True)
    accuracy_score = Column(Float, nullable=True)

    slack_notification_sent = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    vessel = relationship("Vessel", back_populates="predictions")
    terminal = relationship("Terminal", back_populates="predictions")


class TerminalApproachBehavior(Base):
    __tablename__ = "terminal_approach_behaviors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=False, index=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=True, index=True)

    # Pattern data
    approach_distance_km = Column(Float)
    typical_speed_range_min = Column(Float)
    typical_speed_range_max = Column(Float)
    typical_heading_range_min = Column(Float)
    typical_heading_range_max = Column(Float)
    approach_angle_degrees = Column(Float)

    # Statistics
    observation_count = Column(Integer, default=1)
    last_observed = Column(DateTime(timezone=True), server_default=func.now())
    confidence = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    terminal = relationship("Terminal", back_populates="approach_behaviors")
    vessel = relationship("Vessel")


class BehaviorEvent(Base):
    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=False, index=True)
    event_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # Event details
    event_type = Column(String(50), index=True)  # speed_change/course_change/stop/resume
    speed_before = Column(Float)
    speed_after = Column(Float)
    heading_before = Column(Float)
    heading_after = Column(Float)
    magnitude = Column(Float)

    # Context
    nearest_terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=True)
    distance_to_terminal_km = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    vessel = relationship("Vessel", back_populates="behavior_events")
    nearest_terminal = relationship("Terminal")


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_type = Column(String(50), nullable=False, index=True)  # journey/prediction/behavior
    content_id = Column(Integer, nullable=False, index=True)
    text_content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-small dimension
    embedding_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Create indexes
Index('ix_vessels_location', Vessel.current_lat, Vessel.current_lon)
Index('ix_predictions_active', Prediction.vessel_id, Prediction.status)
Index('ix_journeys_completed', VesselJourney.destination_terminal_id, VesselJourney.completed)
Index('ix_behavior_events_vessel_time', BehaviorEvent.vessel_id, BehaviorEvent.event_time.desc())
Index('ix_content_lookup', VectorEmbedding.content_type, VectorEmbedding.content_id)
