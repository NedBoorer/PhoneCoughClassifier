"""
Phone Cough Classifier - Database Models
Tracks calls, classifications, and patient information
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from app.database.database import Base


class CallRecord(Base):
    """Tracks each incoming call"""
    __tablename__ = "call_records"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Call identifiers
    call_sid = Column(String(64), unique=True, index=True)
    caller_number = Column(String(20), index=True)
    twilio_number = Column(String(20))
    
    # Call metadata
    call_status = Column(String(20), default="initiated")  # initiated, recording, processing, completed, failed
    call_duration = Column(Integer, default=0)  # seconds
    language = Column(String(10), default="en")  # en, hi, ta, te, bn, mr, gu, kn, ml, pa
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Audio recording
    recording_url = Column(String(512), nullable=True)
    recording_duration = Column(Float, default=0.0)
    audio_quality_score = Column(Float, nullable=True)  # 0-100
    
    # Patient info (optional intake)
    patient_info = relationship("PatientInfo", back_populates="call", uselist=False)
    
    # Classification result
    classification = relationship("ClassificationResult", back_populates="call", uselist=False)
    
    # Delivery status
    sms_sent = Column(Boolean, default=False)
    sms_delivered_at = Column(DateTime, nullable=True)
    voice_response_given = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<CallRecord(id={self.id}, caller={self.caller_number}, status={self.call_status})>"


class PatientInfo(Base):
    """Optional patient intake information"""
    __tablename__ = "patient_info"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("call_records.id"), unique=True)
    
    # Patient details
    calling_for = Column(String(20), default="self")  # self, family, child, elderly
    age_group = Column(String(20), nullable=True)  # under_18, 18_40, 40_60, over_60
    
    # Cough details
    cough_duration = Column(String(20), nullable=True)  # days, 1_2_weeks, over_2_weeks, over_month
    associated_symptoms = Column(JSON, nullable=True)  # ["fever", "breathing", "chest_pain", "fatigue"]
    
    # Context
    smoking_history = Column(Boolean, nullable=True)
    chronic_conditions = Column(JSON, nullable=True)  # ["asthma", "copd", "diabetes", "heart"]
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    call = relationship("CallRecord", back_populates="patient_info")
    
    def __repr__(self):
        return f"<PatientInfo(id={self.id}, age_group={self.age_group})>"


class ClassificationResult(Base):
    """Cough classification results"""
    __tablename__ = "classification_results"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("call_records.id"), unique=True)
    
    # Classification
    classification = Column(String(50))  # dry, wet, whooping, chronic, normal
    confidence = Column(Float)  # 0.0 - 1.0
    
    # All probabilities
    probabilities = Column(JSON)  # {"dry": 0.8, "wet": 0.1, ...}
    
    # Method used
    method = Column(String(50))  # hear_model, random_forest, rule_based
    model_version = Column(String(20), default="1.0.0")
    
    # Audio features extracted
    audio_features = Column(JSON, nullable=True)
    
    # Timing
    processing_time_ms = Column(Integer)  # milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Recommendation
    severity = Column(String(20))  # mild, moderate, urgent, emergency
    recommendation = Column(Text)
    
    # Relationship
    call = relationship("CallRecord", back_populates="classification")
    
    def __repr__(self):
        return f"<ClassificationResult(id={self.id}, class={self.classification}, conf={self.confidence:.2f})>"


class ModelMetrics(Base):
    """Track model performance over time"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model info
    model_name = Column(String(100))
    model_version = Column(String(20))
    
    # Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Dataset info
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    
    # Timestamps
    trained_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name}, acc={self.accuracy:.2f})>"
