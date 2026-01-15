#!/usr/bin/env python3
"""
Phone Cough Classifier - One-Command Setup
Automates project setup and verification

Usage:
    python setup.py
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(text, status="..."):
    """Print step with status"""
    print(f"  [{status}] {text}")


def print_success(text):
    print_step(text, "✓")


def print_fail(text):
    print_step(text, "✗")


def print_warn(text):
    print_step(text, "!")


def check_python_version():
    """Check Python version >= 3.10"""
    print_step("Checking Python version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_fail(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_directories():
    """Create required directories"""
    print_step("Creating directories")
    
    dirs = ["data", "models", "recordings", "logs"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    
    print_success(f"Created: {', '.join(dirs)}")
    return True


def create_env_file():
    """Create .env from .env.example if not exists"""
    print_step("Setting up .env configuration")
    
    if Path(".env").exists():
        print_success(".env already exists")
        return True
    
    if Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        print_success("Created .env from .env.example")
        print_warn("Please edit .env with your credentials")
    else:
        print_fail(".env.example not found")
        return False
    
    return True


def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing dependencies (this may take a few minutes)")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            check=True,
            capture_output=True
        )
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_fail(f"pip install failed: {e.stderr.decode()[:200]}")
        return False


def initialize_database():
    """Initialize SQLite database"""
    print_step("Initializing database")
    
    try:
        # Use asyncio to run the init
        import asyncio
        
        async def init():
            from app.database.database import init_db
            await init_db()
        
        asyncio.run(init())
        print_success("Database initialized")
        return True
        
    except Exception as e:
        print_warn(f"Database init skipped: {str(e)[:50]}")
        return True  # Non-critical


def train_demo_model():
    """Train a demo model with synthetic data"""
    print_step("Training demo model with synthetic data")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/train_model.py", "--use-synthetic"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print_success("Demo model trained")
            return True
        else:
            print_warn(f"Training skipped: {result.stderr[:100]}")
            return True
            
    except Exception as e:
        print_warn(f"Training skipped: {str(e)[:50]}")
        return True  # Non-critical


def verify_imports():
    """Verify core imports work"""
    print_step("Verifying core imports")
    
    try:
        from app.config import settings
        from app.main import app
        print_success("Core imports OK")
        return True
    except ImportError as e:
        print_fail(f"Import error: {e}")
        return False


def run_health_check():
    """Run a quick health check"""
    print_step("Running health check")
    
    try:
        from app.ml.classifier import get_classifier
        classifier = get_classifier()
        print_success(f"Classifier ready: {classifier.model_type}")
        return True
    except Exception as e:
        print_warn(f"Classifier: {str(e)[:50]}")
        return True


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("""
Next Steps:

1. Edit .env with your credentials:
   - TWILIO_ACCOUNT_SID
   - TWILIO_AUTH_TOKEN
   - TWILIO_PHONE_NUMBER
   - OPENAI_API_KEY

2. Start the server:
   python -m uvicorn app.main:app --reload --port 8000

3. Test the API:
   Open http://localhost:8000/docs

4. For phone calls, expose with ngrok:
   ngrok http 8000
   
5. Configure Twilio webhook:
   Voice webhook: https://your-domain.ngrok.io/twilio/voice/incoming
   (For India: /india/voice/incoming)

Documentation:
   - README.md - Full documentation
   - /docs - API documentation (when server running)
""")


def main():
    """Main setup function"""
    print_header("Phone Cough Classifier Setup")
    print("Building from scratch with real datasets\n")
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    # Run setup steps
    steps = [
        ("Check Python version", check_python_version),
        ("Create directories", create_directories),
        ("Setup configuration", create_env_file),
        ("Install dependencies", install_dependencies),
        ("Verify imports", verify_imports),
        ("Initialize database", initialize_database),
        ("Train demo model", train_demo_model),
        ("Health check", run_health_check),
    ]
    
    failed = False
    for name, func in steps:
        try:
            if not func():
                print_fail(f"{name} - FAILED")
                failed = True
                break
        except Exception as e:
            print_fail(f"{name} - ERROR: {e}")
            failed = True
            break
    
    if not failed:
        print_next_steps()
        return 0
    else:
        print_header("Setup Failed")
        print("Please check the error above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
