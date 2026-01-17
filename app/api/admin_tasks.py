"""
Phone Cough Classifier - Admin Tasks
Background jobs for follow-ups and maintenance
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database.database import get_session
from app.database.models import CallRecord, ClassificationResult
from app.services.twilio_service import get_twilio_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/trigger-followups")
async def trigger_followup_calls(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session)
):
    """
    Cron job endpoint:
    Finds high-risk calls that need follow-up, accounting for farming seasons.

    FARMER-AWARE LOGIC:
    - Off-season (Feb-May): Follow up within 3 days
    - Sowing (Jun-Jul): Follow up within 14 days
    - Growing (Aug-Oct): Normal follow-up (7 days)
    - Harvest (Nov-Jan): Follow up within 21 days
    - URGENT cases override seasonal delays
    """
    from app.config import get_current_farming_season, get_followup_delay_for_season
    from app.database.models import PatientInfo

    logger.info("Running farmer-aware follow-up job...")

    # Get current farming season
    current_season = get_current_farming_season()
    logger.info(f"Current farming season: {current_season}")

    # For urgent cases, always follow up within 24h regardless of season
    urgent_cutoff = datetime.utcnow() - timedelta(hours=24)

    # For non-urgent, use seasonal delays
    season_delay_days = get_followup_delay_for_season(current_season)
    seasonal_cutoff = datetime.utcnow() - timedelta(days=season_delay_days)
    
    # URGENT CASES: Always follow up within 24h (override seasonal delays)
    urgent_stmt = (
        select(CallRecord)
        .join(CallRecord.classification)
        .where(
            and_(
                CallRecord.created_at <= urgent_cutoff,
                (CallRecord.followup_status == "pending") | (CallRecord.followup_status == None),
                ClassificationResult.severity == "urgent"
            )
        )
    )

    # NON-URGENT HIGH-RISK: Use seasonal delays
    seasonal_stmt = (
        select(CallRecord)
        .join(CallRecord.classification)
        .where(
            and_(
                CallRecord.created_at <= seasonal_cutoff,
                (CallRecord.followup_status == "pending") | (CallRecord.followup_status == None),
                ClassificationResult.severity.in_(["high", "severe"])
            )
        )
    )

    # Execute both queries
    urgent_result = await db.execute(urgent_stmt)
    urgent_calls = urgent_result.scalars().all()

    seasonal_result = await db.execute(seasonal_stmt)
    seasonal_calls = seasonal_result.scalars().all()

    # Combine (avoiding duplicates)
    all_calls = list(set(urgent_calls + seasonal_calls))
    logger.info(f"Found {len(urgent_calls)} urgent + {len(seasonal_calls)} seasonal = {len(all_calls)} total calls")
    
    triggered_count = 0
    skipped_harvest = 0
    twilio = get_twilio_service()

    for call in all_calls:
        if not call.caller_number:
            continue

        # Check if this is a farmer in harvest season
        is_farmer = False
        is_harvest_season = (current_season == "harvest")

        # Get patient info to check occupation
        if call.patient_info:
            is_farmer = call.patient_info.occupation in ["farmer", "farm_worker"]

        # If farmer + harvest + non-urgent, send SMS instead of call
        if is_farmer and is_harvest_season and call.classification.severity != "urgent":
            # Send SMS reminder instead of calling
            from app.utils.i18n import get_text
            language = call.language or "en"
            message = get_text("seasonal_followup_message", language)
            message += f"\n\nYour health screening showed {call.classification.severity} risk. "
            message += "Please call back when you have time.\nHealth Helpline: 108"

            sms_sent = twilio.send_sms(call.caller_number, message)
            if sms_sent:
                call.followup_status = "sms_sent"
                call.followup_scheduled_at = datetime.utcnow()
                skipped_harvest += 1
                logger.info(f"Sent SMS to farmer {call.caller_number} (harvest season)")
        else:
            # Normal follow-up call
            logger.info(f"Triggering follow-up for {call.caller_number} (Call: {call.id})")

            # Trigger outbound call
            callback_url = f"{settings.base_url}/admin/voice/outbound-checkup?original_call_sid={call.call_sid}"
            success = twilio.trigger_outbound_call(call.caller_number, callback_url)

            if success:
                call.followup_status = "scheduled"
                call.followup_scheduled_at = datetime.utcnow()
                triggered_count += 1
            else:
                call.followup_status = "failed"

    await db.commit()

    return {
        "status": "success",
        "triggered_calls": triggered_count,
        "sms_reminders": skipped_harvest,
        "current_season": current_season,
        "total_processed": len(all_calls)
    }


@router.post("/voice/outbound-checkup")
async def outbound_checkup_flow(
    original_call_sid: str = None
):
    """
    IVR Flow for the follow-up call.
    """
    from twilio.twiml.voice_response import VoiceResponse, Gather
    from fastapi.responses import Response

    response = VoiceResponse()
    
    # Greeting
    response.say(
        "Namaste. This is your health friend. Yesterday we found some health risks.",
        voice="Polly.Aditi", language="en-IN"
    )
    
    gather = Gather(
        num_digits=1,
        action=f"{settings.base_url}/admin/voice/outbound-checkup/response",
        timeout=10
    )
    gather.say(
        "Did you visit the doctor? Press 1 for Yes. Press 2 if you need the address again.",
        voice="Polly.Aditi", language="en-IN"
    )
    response.append(gather)
    
    response.hangup()
    
    return Response(content=str(response), media_type="application/xml")


@router.post("/voice/outbound-checkup/response")
async def outbound_checkup_response(
    Digits: str = Form(...),
):
    """Handle follow-up response"""
    from twilio.twiml.voice_response import VoiceResponse
    from fastapi.responses import Response

    response = VoiceResponse()
    
    if Digits == '1':
        response.say(
            "That is very good. Please follow the doctor's impact. Stay healthy.",
            voice="Polly.Aditi", language="en-IN"
        )
        # We could update DB here to mark as 'completed_success'
    elif Digits == '2':
        # Send SMS with address (mock)
        response.say(
            "I have sent the clinic address to your phone. Please go today.",
            voice="Polly.Aditi", language="en-IN"
        )
        # Trigger SMS sending logic here if needed
    else:
        response.say("Thank you. Goodbye.", voice="Polly.Aditi", language="en-IN")
        
    return Response(content=str(response), media_type="application/xml")


@router.get("/referrals")
async def get_referrals(db: AsyncSession = Depends(get_session)):
    """Fetch recent referrals with their classification and recording info"""
    stmt = (
        select(CallRecord)
        .join(CallRecord.classification)
        .order_by(CallRecord.created_at.desc())
        .limit(20)
    )
    result = await db.execute(stmt)
    calls = result.scalars().all()
    
    referrals = []
    for call in calls:
        referrals.append({
            "id": call.id,
            "phone": call.caller_number,
            "date": call.created_at.strftime("%Y-%m-%d %H:%M"),
            "classification": call.classification.classification,
            "severity": call.classification.severity,
            "verified": call.classification.visit_verified,
            "recording_url": call.recording_url
        })
    
    return referrals


class VerificationRequest(BaseModel):
    diagnosis: str
    notes: Optional[str] = ""

@router.post("/verify/{call_id}")
async def verify_referral(
    call_id: int, 
    request: VerificationRequest,
    db: AsyncSession = Depends(get_session)
):
    """Mark a referral as verified with doctor's diagnosis (Ground Truth)"""
    stmt = select(ClassificationResult).where(ClassificationResult.call_id == call_id)
    result = await db.execute(stmt)
    classification = result.scalar_one_or_none()
    
    if not classification:
        return {"status": "error", "message": "Record not found"}
        
    classification.visit_verified = True
    classification.visit_verified_at = datetime.utcnow()
    # Store the doctor's diagnosis - this is the Ground Truth for future training!
    classification.verifier_notes = f"Diagnosis: {request.diagnosis} | Notes: {request.notes}"
    
    await db.commit()
    
    return {"status": "success"}


@router.get("/heatmap")
async def get_heatmap_data(db: AsyncSession = Depends(get_session)):
    """
    Aggregate call data by location for the Epidemic Heatmap.
    Returns list of {city, state, count, high_risk_count, lat, lon (mocked for now)}
    """
    from sqlalchemy import func
    
    # Aggregate by city and state
    stmt = (
        select(
            CallRecord.city,
            CallRecord.state,
            func.count(CallRecord.id).label("total_calls"),
            func.sum(
                case(
                    (ClassificationResult.severity.in_(["high", "severe", "urgent"]), 1),
                    else_=0
                )
            ).label("high_risk_calls")
        )
        .join(CallRecord.classification)
        .where(CallRecord.city.isnot(None))
        .group_by(CallRecord.city, CallRecord.state)
    )
    
    # We need 'case' from sqlalchemy
    from sqlalchemy import case
    
    result = await db.execute(stmt)
    rows = result.all()
    
    heatmap_data = []
    
    # Mock coordinates for Indian cities (since we don't have a geocoding DB yet)
    # In a real app, we would use a geocoding service or store lat/lon on write
    city_coords = {
        "Delhi": [28.6139, 77.2090],
        "Mumbai": [19.0760, 72.8777],
        "Bangalore": [12.9716, 77.5946],
        "Chennai": [13.0827, 80.2707],
        "Kolkata": [22.5726, 88.3639],
        "Hyderabad": [17.3850, 78.4867],
        "Pune": [18.5204, 73.8567],
        "Ahmedabad": [23.0225, 72.5714],
        "Jaipur": [26.9124, 75.7873],
        "Lucknow": [26.8467, 80.9462],
        "Patna": [25.5941, 85.1376],
        "Bhopal": [23.2599, 77.4126],
        "Chandigarh": [30.7333, 76.7794],
        "San Francisco": [37.7749, -122.4194], # For testing
        "New York": [40.7128, -74.0060],
        "London": [51.5074, -0.1278],
    }
    
    for row in rows:
        city = row.city
        coords = city_coords.get(city, [20.5937, 78.9629]) # Default to center of India
        
        heatmap_data.append({
            "city": city,
            "state": row.state,
            "count": row.total_calls,
            "high_risk": row.high_risk_calls or 0,
            "lat": coords[0],
            "lng": coords[1]
        })
        
    return heatmap_data

