"""
Phone Cough Classifier - Family Health Dashboard
Endpoints for family group management and collective health tracking
"""
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database.database import get_session
from app.database.models import FamilyGroup, PatientInfo, CallRecord, ClassificationResult
from app.services.twilio_service import get_twilio_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ===============
# Request/Response Models
# ===============

class FamilyStatusResponse(BaseModel):
    """Family health status summary"""
    family_group_id: int
    primary_contact: str
    household_size: int
    members_screened: int
    high_risk_members: int
    last_screening_date: Optional[datetime]
    village: Optional[str]
    asha_worker_assigned: bool
    members: List[dict]


class FamilyMemberSummary(BaseModel):
    """Individual family member summary"""
    patient_id: int
    name: Optional[str]
    relationship: str  # self, spouse, child, parent, etc.
    last_screening: Optional[datetime]
    last_classification: Optional[str]
    risk_level: Optional[str]


# ===============
# Family Dashboard Endpoints
# ===============

@router.get("/status/{phone_number}")
async def get_family_status(
    phone_number: str,
    db: AsyncSession = Depends(get_session)
) -> FamilyStatusResponse:
    """
    Get family health status for a phone number.

    Returns summary of all family members who have been screened
    and their collective health status.
    """
    # Find family group by primary contact
    stmt = select(FamilyGroup).where(
        FamilyGroup.primary_contact_number == phone_number
    )
    result = await db.execute(stmt)
    family_group = result.scalar_one_or_none()

    if not family_group:
        # Check if this number is a member of a family
        stmt = select(PatientInfo).where(
            PatientInfo.family_group_id.isnot(None)
        ).join(CallRecord).where(
            CallRecord.caller_number == phone_number
        )
        result = await db.execute(stmt)
        patient = result.scalar_one_or_none()

        if patient and patient.family_group:
            family_group = patient.family_group
        else:
            raise HTTPException(
                status_code=404,
                detail="No family group found for this number. Start a screening to create one."
            )

    # Get all family members
    stmt = select(PatientInfo).where(
        PatientInfo.family_group_id == family_group.id
    ).join(CallRecord)
    result = await db.execute(stmt)
    members = result.scalars().all()

    # Build member summaries
    member_summaries = []
    for member in members:
        # Get most recent classification
        if member.call and member.call.classification:
            classification = member.call.classification
            member_summaries.append({
                "patient_id": member.id,
                "calling_for": member.calling_for,
                "age_group": member.age_group,
                "last_screening": member.call.created_at,
                "classification": classification.classification,
                "confidence": classification.confidence,
                "severity": classification.severity,
                "occupation": member.occupation,
                "pesticide_exposure": member.pesticide_exposure,
                "dust_exposure": member.dust_exposure
            })

    # Count high-risk members
    high_risk_count = sum(
        1 for m in member_summaries
        if m.get("severity") in ["high", "severe", "urgent"]
    )

    return FamilyStatusResponse(
        family_group_id=family_group.id,
        primary_contact=family_group.primary_contact_number,
        household_size=family_group.household_size,
        members_screened=len(member_summaries),
        high_risk_members=high_risk_count,
        last_screening_date=family_group.last_screening_date,
        village=family_group.village,
        asha_worker_assigned=family_group.asha_worker_assigned,
        members=member_summaries
    )


@router.post("/link-member")
async def link_family_member(
    primary_contact: str,
    member_phone: str,
    relationship: str = "family",
    db: AsyncSession = Depends(get_session)
):
    """
    Link a new family member to an existing family group.

    Used when a family member calls independently but should be
    linked to an existing family health record.
    """
    # Find or create family group
    stmt = select(FamilyGroup).where(
        FamilyGroup.primary_contact_number == primary_contact
    )
    result = await db.execute(stmt)
    family_group = result.scalar_one_or_none()

    if not family_group:
        # Create new family group
        family_group = FamilyGroup(
            primary_contact_number=primary_contact,
            household_size=1,
            created_at=datetime.utcnow()
        )
        db.add(family_group)
        await db.flush()

    # Find the member's most recent call
    stmt = select(CallRecord).where(
        CallRecord.caller_number == member_phone
    ).order_by(CallRecord.created_at.desc()).limit(1)
    result = await db.execute(stmt)
    call_record = result.scalar_one_or_none()

    if not call_record:
        raise HTTPException(
            status_code=404,
            detail=f"No call record found for {member_phone}"
        )

    # Find or create patient info
    if not call_record.patient_info:
        patient_info = PatientInfo(
            call_id=call_record.id,
            calling_for=relationship,
            family_group_id=family_group.id
        )
        db.add(patient_info)
    else:
        # Update existing patient info
        call_record.patient_info.family_group_id = family_group.id
        call_record.patient_info.calling_for = relationship

    # Update family group metadata
    family_group.household_size = await db.scalar(
        select(func.count(PatientInfo.id)).where(
            PatientInfo.family_group_id == family_group.id
        )
    )
    family_group.last_screening_date = datetime.utcnow()

    await db.commit()

    return {
        "status": "success",
        "family_group_id": family_group.id,
        "member_linked": member_phone,
        "household_size": family_group.household_size
    }


@router.post("/send-family-report")
async def send_family_health_report(
    family_group_id: int,
    send_to_number: Optional[str] = None,
    db: AsyncSession = Depends(get_session)
):
    """
    Send SMS summary of entire family's health status.

    Useful for ASHA workers or family heads to get overview.
    """
    # Get family group
    stmt = select(FamilyGroup).where(FamilyGroup.id == family_group_id)
    result = await db.execute(stmt)
    family_group = result.scalar_one_or_none()

    if not family_group:
        raise HTTPException(status_code=404, detail="Family group not found")

    # Get all members with their latest screenings
    stmt = select(PatientInfo).where(
        PatientInfo.family_group_id == family_group_id
    ).join(CallRecord).join(ClassificationResult, isouter=True)
    result = await db.execute(stmt)
    members = result.scalars().all()

    # Build SMS message
    message = f"🏠 Family Health Report\n"
    message += f"Family: {family_group.primary_contact_number}\n"
    if family_group.village:
        message += f"Village: {family_group.village}\n"
    message += f"\nMembers: {len(members)}\n\n"

    high_risk = []
    moderate_risk = []
    low_risk = []

    for member in members:
        if member.call and member.call.classification:
            name = member.calling_for or "Member"
            severity = member.call.classification.severity
            classification = member.call.classification.classification

            risk_emoji = {
                "mild": "🟢", "low": "🟢", "normal": "🟢",
                "moderate": "🟡",
                "high": "🔴", "severe": "🔴", "urgent": "🚨"
            }
            emoji = risk_emoji.get(severity, "⚪")

            member_line = f"{emoji} {name.title()}: {classification}"

            if severity in ["high", "severe", "urgent"]:
                high_risk.append(member_line)
            elif severity == "moderate":
                moderate_risk.append(member_line)
            else:
                low_risk.append(member_line)

    # Prioritize high-risk members
    if high_risk:
        message += "⚠️ URGENT ATTENTION:\n" + "\n".join(high_risk) + "\n\n"
    if moderate_risk:
        message += "⚠️ Monitor:\n" + "\n".join(moderate_risk) + "\n\n"
    if low_risk:
        message += "✓ Normal:\n" + "\n".join(low_risk) + "\n\n"

    message += "\nFor detailed results, call the health line.\n"
    message += "Health Helpline: 108"

    # Send SMS
    recipient = send_to_number or family_group.primary_contact_number
    twilio = get_twilio_service()
    success = twilio.send_sms(recipient, message)

    if success:
        return {
            "status": "success",
            "message": "Family report sent",
            "sent_to": recipient,
            "members_included": len(members)
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to send SMS report"
        )


@router.get("/asha-dashboard/{asha_worker_id}")
async def get_asha_dashboard(
    asha_worker_id: str,
    db: AsyncSession = Depends(get_session)
):
    """
    Dashboard for ASHA workers showing all families they manage.

    Returns summary statistics and list of families needing attention.
    """
    # Get all families assigned to this ASHA worker
    stmt = select(FamilyGroup).where(
        FamilyGroup.asha_worker_id == asha_worker_id
    )
    result = await db.execute(stmt)
    families = result.scalars().all()

    if not families:
        return {
            "asha_worker_id": asha_worker_id,
            "total_families": 0,
            "message": "No families assigned yet"
        }

    # Calculate statistics
    total_families = len(families)
    total_members = sum(f.household_size for f in families)
    families_with_high_risk = sum(1 for f in families if f.has_high_risk_member)

    # Get families needing attention (high risk or no recent screening)
    needs_attention = []
    from datetime import timedelta
    one_month_ago = datetime.utcnow() - timedelta(days=30)

    for family in families:
        if family.has_high_risk_member:
            needs_attention.append({
                "family_id": family.id,
                "contact": family.primary_contact_number,
                "village": family.village,
                "reason": "High-risk member",
                "last_screening": family.last_screening_date
            })
        elif not family.last_screening_date or family.last_screening_date < one_month_ago:
            needs_attention.append({
                "family_id": family.id,
                "contact": family.primary_contact_number,
                "village": family.village,
                "reason": "Overdue for screening",
                "last_screening": family.last_screening_date
            })

    return {
        "asha_worker_id": asha_worker_id,
        "total_families": total_families,
        "total_members": total_members,
        "families_with_high_risk": families_with_high_risk,
        "needs_attention": needs_attention,
        "coverage_percentage": f"{(total_members / (total_families * 4)) * 100:.1f}%"  # Assuming avg 4 members/family
    }
