import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.referral_service import ReferralService
from app.database.models import ClassificationResult, CallRecord

def test_referral_verification_success():
    """Test successful referral verification and notification"""
    asyncio.run(_test_referral_verification_success_async())

async def _test_referral_verification_success_async():
    
    # Mock mocks
    mock_db_session = AsyncMock()
    mock_twilio = MagicMock()
    
    # Mock Data
    referral_code = "REF-1234"
    caller_number = "+919999999999"
    verifier_number = "+918888888888"
    
    mock_call_record = CallRecord(caller_number=caller_number, language="en")
    mock_classification = ClassificationResult(
        referral_code=referral_code, 
        visit_verified=False,
        call=mock_call_record
    )
    
    # Mock DB Query Result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_classification
    mock_db_session.execute.return_value = mock_result
    
    # Patch dependencies
    with patch("app.services.referral_service.async_session_maker") as mock_maker, \
         patch("app.services.referral_service.get_twilio_service", return_value=mock_twilio):
        
        mock_maker.return_value.__aenter__.return_value = mock_db_session
        
        service = ReferralService()
        success = await service.verify_referral(referral_code, verifier_number)
        
        assert success is True
        assert mock_classification.visit_verified is True
        assert mock_classification.visit_verified_by == verifier_number
        assert mock_classification.visit_verified_at is not None
        
        # Verify Notification
        mock_twilio.send_whatsapp.assert_called()
        args = mock_twilio.send_whatsapp.call_args[0]
        assert args[0] == caller_number
        assert "REF-1234" in args[1]

def test_referral_not_found():
    """Test verification failures for invalid code"""
    asyncio.run(_test_referral_not_found_async())

async def _test_referral_not_found_async():
    
    mock_db_session = AsyncMock()
    mock_twilio = MagicMock()
    
    # Mock Empty Result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result
    
    with patch("app.services.referral_service.async_session_maker") as mock_maker, \
         patch("app.services.referral_service.get_twilio_service", return_value=mock_twilio):
        
        mock_maker.return_value.__aenter__.return_value = mock_db_session
        
        service = ReferralService()
        success = await service.verify_referral("INVALID-CODE", "+918888888888")
        
        assert success is False
        mock_twilio.send_whatsapp.assert_not_called()
