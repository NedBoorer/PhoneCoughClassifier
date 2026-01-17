"""
Medical Report Endpoints
API endpoints for generating and sending doctor-friendly medical reports
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.twilio_service import get_twilio_service, format_medical_report
from app.database.database import async_session_maker
from app.database.models import CallRecord, ClassificationResult

router = APIRouter(prefix="/medical-report", tags=["Medical Reports"])
logger = logging.getLogger(__name__)


class MedicalReportRequest(BaseModel):
    """Request for medical report generation"""
    call_sid: Optional[str] = None
    phone_number: Optional[str] = None
    report_type: str = "brief"  # "brief", "summary", or "detailed"
    send_sms: bool = True


class MedicalReportResponse(BaseModel):
    """Response containing medical report"""
    success: bool
    report: str
    message: str


@router.post("/generate", response_model=MedicalReportResponse)
async def generate_medical_report_endpoint(
    request: MedicalReportRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a medical report from a previous screening.
    
    Users can request this via SMS: "Send report" or "Medical report"
    Report will be texted back in doctor-friendly format.
    """
    try:
        # Find the most recent screening for this user
        async with async_session_maker() as db:
            query = db.query(CallRecord).join(ClassificationResult)
            
            if request.call_sid:
                query = query.filter(CallRecord.call_sid == request.call_sid)
            elif request.phone_number:
                query = query.filter(CallRecord.caller_number == request.phone_number)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either call_sid or phone_number required"
                )
            
            call_record = query.order_by(CallRecord.created_at.desc()).first()
            
            if not call_record:
                raise HTTPException(
                    status_code=404,
                    detail="No screening found for this user"
                )
            
            # Get classification result
            classification = call_record.classification_results[0] if call_record.classification_results else None
            
            if not classification:
                raise HTTPException(
                    status_code=404,
                    detail="No analysis results found"
                )
        
        # Reconstruct comprehensive result from database
        from app.ml.model_hub import ComprehensiveHealthResult, ScreeningResult
        
        # Build screening results from stored data
        screenings = {}
        
        # Main classification (respiratory/cough)
        if classification.classification:
            screenings["respiratory"] = ScreeningResult(
                disease="respiratory",
                detected=classification.severity not in ["normal", "low"],
                confidence=classification.confidence,
                severity=classification.severity,
                details={
                    "sound_class": classification.classification,
                    "probabilities": classification.probabilities or {}
                },
                recommendation=classification.recommendation or ""
            )
        
        # Reconstruct comprehensive result
        comprehensive_result = ComprehensiveHealthResult(
            primary_concern=classification.classification or "unknown",
            overall_risk_level=classification.severity or "low",
            screenings=screenings,
            voice_biomarkers={},
            processing_time_ms=classification.processing_time_ms or 0,
            recommendation=classification.recommendation or "Please consult a doctor for proper evaluation."
        )
        
        # Generate medical report
        report = format_medical_report(
            comprehensive_result=comprehensive_result,
            patient_phone=call_record.caller_number,
            report_type=request.report_type
        )
        
        # Send via SMS if requested
        if request.send_sms and call_record.caller_number:
            background_tasks.add_task(
                _send_medical_report_sms,
                call_record.caller_number,
                report,
                request.report_type
            )
            message = f"Medical report generated and sent to {call_record.caller_number}"
        else:
            message = "Medical report generated successfully"
        
        return MedicalReportResponse(
            success=True,
            report=report,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate medical report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


def _send_medical_report_sms(phone_number: str, report: str, report_type: str):
    """Background task to send medical report via SMS"""
    try:
        twilio_service = get_twilio_service()
        
        # For brief/detailed reports, split into multiple SMS if needed
        if report_type in ["brief", "detailed"]:
            # Split into SMS-sized chunks (1600 chars per message)
            max_length = 1600
            if len(report) <= max_length:
                twilio_service.send_sms(phone_number, report)
            else:
                # Split into multiple messages
                parts = []
                lines = report.split('\n')
                current_part = []
                current_length = 0
                
                for line in lines:
                    line_length = len(line) + 1  # +1 for newline
                    if current_length + line_length > max_length:
                        parts.append('\n'.join(current_part))
                        current_part = [line]
                        current_length = line_length
                    else:
                        current_part.append(line)
                        current_length += line_length
                
                if current_part:
                    parts.append('\n'.join(current_part))
                
                # Send each part
                for i, part in enumerate(parts, 1):
                    header = f"[Medical Report {i}/{len(parts)}]\n\n" if len(parts) > 1 else ""
                    twilio_service.send_sms(phone_number, header + part)
                    logger.info(f"Sent medical report part {i}/{len(parts)} to {phone_number}")
        else:
            # Summary - single SMS
            twilio_service.send_sms(phone_number, report)
        
        logger.info(f"Medical report sent to {phone_number}")
        
    except Exception as e:
        logger.error(f"Failed to send medical report SMS: {e}")


@router.get("/latest/{phone_number}", response_model=MedicalReportResponse)
async def get_latest_medical_report(
    phone_number: str,
    report_type: str = "brief",
    send_sms: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Get the latest medical report for a phone number.
    
    Convenience endpoint for users to retrieve their report.
    """
    request = MedicalReportRequest(
        phone_number=phone_number,
        report_type=report_type,
        send_sms=send_sms
    )
    
    return await generate_medical_report_endpoint(request, background_tasks)
