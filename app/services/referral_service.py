"""
Phone Cough Classifier - Referral Service
Handles the "Digital Handshake" loop closure:
- Verifying patient visits via SMS from doctors
- Updating referral status
- Notifying ASHA workers
"""
import logging
from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.database.database import async_session_maker
from app.database.models import ClassificationResult, CallRecord, FamilyGroup
from app.services.twilio_service import get_twilio_service

logger = logging.getLogger(__name__)

class ReferralService:
    def __init__(self):
        self.twilio_service = get_twilio_service()

    async def verify_referral(self, referral_code: str, verifier_phone: str, notes: str = None) -> bool:
        """
        Verify a visit based on referral code and notify the original referrer (ASHA/Family).
        Returns True if successful, False if code not found.
        """
        logger.info(f"Verifying referral {referral_code} from {verifier_phone}")
        
        async with async_session_maker() as db:
            # 1. Find the referral
            result = await db.execute(
                select(ClassificationResult)
                .options(selectinload(ClassificationResult.call))
                .where(ClassificationResult.referral_code == referral_code)
            )
            classification = result.scalar_one_or_none()
            
            if not classification:
                logger.warning(f"Referral code {referral_code} not found")
                return False
                
            # 2. Update status
            if classification.visit_verified:
                 logger.info(f"Referral {referral_code} already verified")
                 # We still return True to confirm to the doctor "Yes, it's done"
                 return True

            classification.visit_verified = True
            classification.visit_verified_at = datetime.utcnow()
            classification.visit_verified_by = verifier_phone
            classification.verifier_notes = notes
            
            await db.commit()
            
            # 3. Notify ASHA / Primary Contact
            await self._notify_referrer(classification, db)
            
            return True

    async def _notify_referrer(self, classification: ClassificationResult, db):
        """Send WhatsApp/SMS to the person who originated the referral (ASHA or Family Head)"""
        try:
            call_record = classification.call
            if not call_record:
                logger.warning("No call record associated with classification")
                return

            original_caller = call_record.caller_number
            language = call_record.language or 'en'
            
            # Check if this was an ASHA call (could check PatientInfo or just assume caller is the contact)
            # Ideally we check if caller is an ASHA, but for MVP we notify the phone that made the call.
            
            status_msg = f"✅ Success! Patient with ticket {classification.referral_code} has been seen by a doctor."
            
            if language != 'en':
                # Simple Hindi fallback for MVP or other indic langs
                status_msg = f"✅ सफल! टिकट {classification.referral_code} वाले मरीज को डॉक्टर ने देख लिया है।"

            # Send WhatsApp if possible, else SMS
            # For ASHA workers, we prefer WhatsApp
            try:
                self.twilio_service.send_whatsapp(original_caller, status_msg)
            except Exception:
                # Fallback to SMS
                self.twilio_service.send_sms(original_caller, status_msg)
                
            logger.info(f"Notified referrer {original_caller} of successful visit")

        except Exception as e:
            logger.error(f"Failed to notify referrer: {e}")

def get_referral_service():
    return ReferralService()
