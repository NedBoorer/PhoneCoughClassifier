
import pytest
from fastapi.testclient import TestClient
import xml.etree.ElementTree as ET
from app.main import app
from app.config import settings

client = TestClient(app)

def parse_twiml(content):
    """Helper to parse TwiML XML"""
    return ET.fromstring(content)

def test_india_incoming_call_menu():
    """Test the main language menu plays prompts for all languages"""
    response = client.post(
        "/india/voice/incoming",
        data={
            "CallSid": "test_sid",
            "From": "+911234567890",
            "To": settings.twilio_phone_number
        }
    )
    assert response.status_code == 200
    
    root = parse_twiml(response.text)
    
    # Needs to find <Gather>
    gather = root.find("Gather")
    assert gather is not None
    
    # Check for multiple <Say> verbs inside Gather
    says = gather.findall("Say")
    
    # We expect at least one prompt for each of the 10 languages + ASHA
    # The prompts are added in order: En, Hi, Ta, Te, Bn, Mr, Gu, Kn, Ml, Pa, ASHA
    assert len(says) >= 11
    
    # Check voices
    # Google voices usually start with 'Google.'
    voices = [say.get("voice") for say in says]
    
    # Ensure regional languages utilize Google voices
    assert any("Google.ta-IN" in v for v in voices) # Tamil
    assert any("Google.te-IN" in v for v in voices) # Telugu
    assert any("Google.pa-IN" in v for v in voices) # Punjabi
    
    # Verify ASHA prompt mentions "star"
    asha_prompt = says[-1].text
    assert "star" in asha_prompt.lower()

def test_language_selection_tamil():
    """Test selecting Tamil (Option 3)"""
    response = client.post(
        "/india/voice/language-selected",
        data={
            "CallSid": "test_sid",
            "From": "+911234567890",
            "Digits": "3" # Tamil
        }
    )
    assert response.status_code == 200
    root = parse_twiml(response.text)
    
    # Check redirect to start recording with lang=ta
    redirect = root.find("Redirect")
    assert redirect is not None
    assert "lang=ta" in redirect.text

def test_language_selection_punjabi():
    """Test selecting Punjabi (Option 0)"""
    response = client.post(
        "/india/voice/language-selected",
        data={
            "CallSid": "test_sid",
            "From": "+911234567890",
            "Digits": "0" # Punjabi
        }
    )
    assert response.status_code == 200
    root = parse_twiml(response.text)
    
    # Check redirect to start recording with lang=pa
    redirect = root.find("Redirect")
    assert redirect is not None
    assert "lang=pa" in redirect.text

def test_asha_mode_selection():
    """Test selecting ASHA Mode (Option *)"""
    response = client.post(
        "/india/voice/language-selected",
        data={
            "CallSid": "test_sid",
            "From": "+911234567890",
            "Digits": "*" # Star Key
        }
    )
    assert response.status_code == 200
    root = parse_twiml(response.text)
    
    # Check redirect to ASHA menu
    redirect = root.find("Redirect")
    assert redirect is not None
    assert "/india/voice/asha/menu" in redirect.text
