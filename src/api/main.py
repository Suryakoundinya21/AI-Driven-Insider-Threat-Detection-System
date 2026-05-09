from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.data_store     import load_data, get_df, get_alert_df
from src.api.routes_alerts  import router as alerts_router
from src.api.routes_users   import router as users_router
from src.api.routes_stats   import router as stats_router
from src.api.routes_feedback import router as feedback_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading data...")
    load_data()
    logger.info("Data loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Insider Threat Detection API",
    description = "AI-powered insider threat detection with explainable alerts",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(alerts_router)
app.include_router(users_router)
app.include_router(stats_router)
app.include_router(feedback_router)


@app.get("/", tags=["Health"])
def root():
    df       = get_df()
    alert_df = get_alert_df()
    return {
        "status"         : "running",
        "service"        : "Insider Threat Detection API",
        "version"        : "2.0.0",
        "total_sessions" : len(df) if df is not None else 0,
        "total_alerts"   : len(alert_df) if alert_df is not None else 0,
        "endpoints"      : {
            "alerts"   : "/alerts/",
            "users"    : "/users/top-risk",
            "stats"    : "/stats/overview",
            "feedback" : "/feedback/stats",
            "docs"     : "/docs",
        }
    }


@app.get("/health", tags=["Health"])
def health():
    df = get_df()
    return {
        "status"      : "healthy" if df is not None else "loading",
        "data_loaded" : df is not None,
    }
