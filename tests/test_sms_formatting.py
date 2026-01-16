
import pytest
from app.services.twilio_service import format_sms_result
from app.utils.i18n import get_text

def test_sms_format_english():
    msg = format_sms_result(
        classification="dry",
        confidence=0.95,
        recommendation="Drink water.",
        language="en"
    )
    assert "Cough Analysis Result" in msg
    assert "Classification: Dry" in msg
    assert "Confidence: 95%" in msg
    assert "Drink water." in msg
    assert "Health Helpline" in msg

def test_sms_format_hindi():
    msg = format_sms_result(
        classification="wet",
        confidence=0.80,
        recommendation="Garam paani.",
        language="hi"
    )
    assert "खांसी विश्लेषण परिणाम" in msg
    assert "वर्गीकरण: बलगम वाली खांसी" in msg
    assert "विश्वास स्तर: 80%" in msg
    assert "Garam paani." in msg
    assert "हेल्पलाइन" in msg

def test_sms_format_tamil():
    msg = format_sms_result(
        classification="whooping",
        confidence=0.88,
        recommendation="Rest well.",
        language="ta"
    )
    assert "இருமல் பகுப்பாய்வு முடிவு" in msg # Result title
    assert "வகைப்பாடு: கக்குவான் இருமல்" in msg # Classification: Whooping
    assert "நம்பிக்கை: 88%" in msg # Confidence
    assert "சுகாதார உதவி எண்" in msg # Helpline

def test_sms_format_punjabi():
    msg = format_sms_result(
        classification="normal",
        confidence=0.99,
        recommendation="All good.",
        language="pa"
    )
    assert "ਖੰਘ ਵਿਸ਼ਲੇਸ਼ਣ ਨਤੀਜਾ" in msg
    assert "ਵਰਗੀਕਰਨ: ਆਮ ਖੰਘ" in msg
    assert "ਭਰੋਸਾ: 99%" in msg
    assert "ਸਿਹਤ ਹੈਲਪਲਾਈਨ" in msg
