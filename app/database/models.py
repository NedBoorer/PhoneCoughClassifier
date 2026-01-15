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
    
    # Engagement & Action
    followup_status = Column(String(20), default="pending")  # pending, scheduled, completed, failed
    followup_scheduled_at = Column(DateTime, nullable=True)
    followup_attempted_at = Column(DateTime, nullable=True)
    
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

    # FARMER-SPECIFIC FIELDS
    occupation = Column(String(30), nullable=True)  # farmer, farm_worker, asha_worker, other
    pesticide_exposure = Column(Boolean, nullable=True)  # Occupational hazard for farmers
    dust_exposure = Column(Boolean, nullable=True)  # Crop dust, hay, grain dust
    work_environment = Column(String(20), nullable=True)  # outdoor, indoor, mixed
    farming_season = Column(String(20), nullable=True)  # sowing, growing, harvest, off_season
    family_group_id = Column(Integer, ForeignKey("family_groups.id"), nullable=True)  # Link to family

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    call = relationship("CallRecord", back_populates="patient_info")
    family_group = relationship("FamilyGroup", back_populates="members")
    
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
    
    # Action
    referral_code = Column(String(20), nullable=True)  # The "Golden Ticket"

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


class HealthAssessment(Base):
    """Health assessment results for Parkinson's and Depression screening"""
    __tablename__ = "health_assessments"

    id = Column(Integer, primary_key=True, index=True)

    # Call identifiers
    call_sid = Column(String(64), index=True)
    caller_number = Column(String(20), index=True)

    # Assessment type
    assessment_type = Column(String(20), index=True)  # 'parkinsons' or 'depression'

    # Classification results
    classification = Column(String(50))  # risk_level for PD, severity_level for depression
    confidence = Column(Float)  # 0.0 - 1.0
    risk_level = Column(String(30))  # normalized risk/severity level

    # Detailed indicators
    indicators = Column(JSON)  # feature values and indicator levels

    # Processing metadata
    method = Column(String(50))  # sklearn, rule_based
    processing_time_ms = Column(Integer)

    # Recommendation
    recommendation = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<HealthAssessment(id={self.id}, type={self.assessment_type}, result={self.classification})>"


class FamilyGroup(Base):
    """
    Family group tracking for rural households.
    Enables screening entire families and tracking health trends.
    """
    __tablename__ = "family_groups"

    id = Column(Integer, primary_key=True, index=True)

    # Primary contact (usually head of household or ASHA worker)
    primary_contact_number = Column(String(20), index=True)
    primary_contact_name = Column(String(100), nullable=True)

    # Location (village/district for regional health tracking)
    village = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)

    # Family metadata
    household_size = Column(Integer, default=1)
    asha_worker_assigned = Column(Boolean, default=False)
    asha_worker_id = Column(String(50), nullable=True)  # ASHA worker identification

    # Risk tracking
    has_high_risk_member = Column(Boolean, default=False)
    last_screening_date = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    members = relationship("PatientInfo", back_populates="family_group")

    def __repr__(self):
        return f"<FamilyGroup(id={self.id}, contact={self.primary_contact_number}, size={self.household_size})>"
