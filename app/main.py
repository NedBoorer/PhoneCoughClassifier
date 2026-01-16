"""
Phone Cough Classifier - FastAPI Main Application
Voice agent pipeline for cough classification using real COUGHVID dataset
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Phone Cough Classifier...")
    logger.info(f"   Environment: {settings.environment}")
    logger.info(f"   Base URL: {settings.base_url}")
    
    # Create directories
    settings.data_dir
    settings.models_dir
    settings.recordings_dir
    
    # Initialize database
    try:
        from app.database.database import init_db
        await init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.warning(f"Database init skipped: {e}")
    
    # Load ML model
    try:
        from app.ml.classifier import get_classifier
        classifier = get_classifier()
        logger.info(f"✓ ML cough classifier loaded: {classifier.model_type}")
    except Exception as e:
        logger.warning(f"ML cough model not loaded: {e}")
    
    # Load health assessment classifiers
    try:
        from app.ml.parkinsons_classifier import get_parkinsons_classifier
        pd_classifier = get_parkinsons_classifier()
        logger.info(f"✓ Parkinson's classifier loaded: {pd_classifier.model_type}")
    except Exception as e:
        logger.warning(f"Parkinson's classifier not loaded: {e}")
    
    try:
        from app.ml.depression_classifier import get_depression_classifier
        dep_classifier = get_depression_classifier()
        logger.info(f"✓ Depression classifier loaded: {dep_classifier.model_type}")
    except Exception as e:
        logger.warning(f"Depression classifier not loaded: {e}")
    
    # Initialize RAG knowledge base for voice agent
    if settings.enable_voice_agent:
        try:
            from app.services.rag_service import initialize_rag_service
            await initialize_rag_service()
            logger.info("✓ RAG knowledge base initialized")
        except Exception as e:
            logger.warning(f"RAG knowledge base not initialized: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Phone Cough Classifier...")
    try:
        from app.database.database import close_db
        await close_db()
    except Exception as e:
        logger.error(f"Error during database shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Phone Cough Classifier",
    description="Voice agent pipeline for cough classification using COUGHVID dataset",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================
# Core Endpoints
# ==================

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Phone Cough Classifier",
        "version": "1.0.0",
        "description": "Voice agent for cough classification",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = {
        "status": "healthy",
        "environment": settings.environment,
        "components": {}
    }
    
    # Check database
    try:
        from app.database.database import async_session_maker
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
        health["components"]["database"] = "connected"
    except Exception as e:
        health["components"]["database"] = f"error: {str(e)}"
    
    # Check ML model
    try:
        from app.ml.classifier import get_classifier
        classifier = get_classifier()
        health["components"]["cough_classifier"] = classifier.model_type
    except Exception as e:
        health["components"]["cough_classifier"] = f"not loaded: {str(e)}"
    
    # Check Parkinson's classifier
    try:
        from app.ml.parkinsons_classifier import get_parkinsons_classifier
        pd_classifier = get_parkinsons_classifier()
        health["components"]["parkinsons_classifier"] = pd_classifier.model_type
    except Exception as e:
        health["components"]["parkinsons_classifier"] = f"not loaded: {str(e)}"
    
    # Check Depression classifier
    try:
        from app.ml.depression_classifier import get_depression_classifier
        dep_classifier = get_depression_classifier()
        health["components"]["depression_classifier"] = dep_classifier.model_type
    except Exception as e:
        health["components"]["depression_classifier"] = f"not loaded: {str(e)}"
    
    # Check Twilio config
    health["components"]["twilio"] = "configured" if settings.twilio_account_sid else "not configured"
    
    # Check OpenAI config
    health["components"]["openai"] = "configured" if settings.openai_api_key else "not configured"
    
    return health


# ==================
# Include Routers
# ==================

# Twilio webhooks
try:
    from app.api.twilio_webhooks import router as twilio_router
    app.include_router(twilio_router, prefix="/twilio", tags=["Twilio Webhooks"])
    logger.info("✓ Twilio webhooks loaded")
except ImportError as e:
    logger.warning(f"Twilio webhooks not loaded: {e}")

# India accessibility webhooks
try:
    from app.api.india_webhooks import router as india_router
    app.include_router(india_router, prefix="/india", tags=["India Accessibility"])
    logger.info("✓ India webhooks loaded")
except ImportError as e:
    logger.warning(f"India webhooks not loaded: {e}")

# Test endpoints
try:
    from app.api.test_endpoints import router as test_router
    app.include_router(test_router, prefix="/test", tags=["Testing"])
    logger.info("✓ Test endpoints loaded")
except ImportError as e:
    logger.warning(f"Test endpoints not loaded: {e}")

# Health assessment webhooks (Parkinson's, Depression)
try:
    from app.api.health_webhooks import router as health_router
    app.include_router(health_router, prefix="/health", tags=["Health Assessment"])
    logger.info("✓ Health assessment webhooks loaded")
except ImportError as e:
    logger.warning(f"Health webhooks not loaded: {e}")

# Admin & Background Tasks
try:
    from app.api.admin_tasks import router as admin_router
    app.include_router(admin_router, prefix="/admin", tags=["Admin Tasks"])
    logger.info("✓ Admin tasks loaded")
except ImportError as e:
    logger.warning(f"Admin tasks not loaded: {e}")

# Family Health Dashboard
try:
    from app.api.family_endpoints import router as family_router
    app.include_router(family_router, prefix="/family", tags=["Family Health"])
    logger.info("✓ Family endpoints loaded")
except ImportError as e:
    logger.warning(f"Family endpoints not loaded: {e}")

# Voice Agent (Conversational AI)
if settings.enable_voice_agent:
    try:
        from app.api.voice_agent_webhooks import router as voice_agent_router
        app.include_router(voice_agent_router, prefix="/voice-agent", tags=["Voice Agent"])
        logger.info("✓ Voice agent webhooks loaded")
    except ImportError as e:
        logger.warning(f"Voice agent webhooks not loaded: {e}")


# ==================
# Static Files
# ==================
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Mount Data for accessible images (WhatsApp)
app.mount("/data", StaticFiles(directory=str(settings.data_dir)), name="data")
app.mount("/recordings", StaticFiles(directory=str(settings.recordings_dir)), name="recordings")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
