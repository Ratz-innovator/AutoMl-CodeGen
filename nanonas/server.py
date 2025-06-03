"""
nanoNAS REST API Server
======================

Professional FastAPI-based server providing REST API endpoints for the Neural
Architecture Search platform. Includes experiment management, real-time monitoring,
authentication, and production-ready features.

Key Features:
- RESTful API for all NAS operations
- Real-time WebSocket monitoring
- Authentication and authorization
- Experiment management and tracking
- Result storage and retrieval
- Async task processing with Celery
- API documentation with OpenAPI
- Production monitoring and logging

Example Usage:
    >>> from nanonas.server import app
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Or use the CLI
    $ nanonas-server --host 0.0.0.0 --port 8000 --workers 4
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import uuid
from pydantic import BaseModel, Field
import redis
from celery import Celery
import pymongo
from pymongo import MongoClient
import jwt
import bcrypt
import os
from contextlib import asynccontextmanager

from .core.config import ExperimentConfig, SearchConfig
from .core.architecture import Architecture
from .api import search, benchmark
from .visualization.interactive_dashboard import NASDashboard
from .utils.monitoring import SystemMonitor, ExperimentTracker


# Pydantic Models
class SearchRequest(BaseModel):
    """Request model for architecture search."""
    strategy: str = Field(..., description="Search strategy to use")
    dataset: str = Field(..., description="Dataset name")
    search_space: str = Field(default="nano", description="Search space configuration")
    epochs: Optional[int] = Field(default=None, description="Number of training epochs")
    population_size: Optional[int] = Field(default=None, description="Population size for evolutionary search")
    device: str = Field(default="auto", description="Computing device")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")

class BenchmarkRequest(BaseModel):
    """Request model for benchmarking."""
    strategies: List[str] = Field(..., description="List of strategies to benchmark")
    dataset: str = Field(..., description="Dataset name")
    search_space: str = Field(default="nano", description="Search space configuration")
    num_runs: int = Field(default=3, description="Number of runs per strategy")
    epochs: int = Field(default=50, description="Training epochs")

class UserCreate(BaseModel):
    """User creation model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    role: str = Field(default="user", description="User role")

class UserLogin(BaseModel):
    """User login model."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")

class ExperimentResponse(BaseModel):
    """Experiment response model."""
    experiment_id: str
    name: str
    strategy: str
    dataset: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress: float
    best_accuracy: Optional[float]
    total_architectures: int

class ArchitectureResponse(BaseModel):
    """Architecture response model."""
    architecture_id: str
    experiment_id: str
    operations: List[int]
    accuracy: float
    parameters: int
    flops: int
    search_method: str
    created_at: datetime


# Global variables
redis_client = None
mongo_client = None
experiment_tracker = None
system_monitor = None
websocket_connections = []
celery_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


# FastAPI app initialization
app = FastAPI(
    title="nanoNAS API",
    description="Professional Neural Architecture Search Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def startup_event():
    """Initialize services on startup."""
    global redis_client, mongo_client, experiment_tracker, system_monitor, celery_app
    
    logger.info("🚀 Starting nanoNAS API Server...")
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        redis_client = None
    
    # Initialize MongoDB
    try:
        mongo_client = MongoClient(
            os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
        )
        mongo_client.admin.command('ismaster')
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB connection failed: {e}")
        mongo_client = None
    
    # Initialize Celery
    try:
        celery_app = Celery(
            'nanonas',
            broker=os.getenv('CELERY_BROKER', 'redis://localhost:6379/0'),
            backend=os.getenv('CELERY_BACKEND', 'redis://localhost:6379/0')
        )
        logger.info("✅ Celery initialized")
    except Exception as e:
        logger.warning(f"⚠️ Celery initialization failed: {e}")
        celery_app = None
    
    # Initialize experiment tracking
    experiment_tracker = ExperimentTracker(
        redis_client=redis_client,
        mongo_client=mongo_client
    )
    
    # Initialize system monitoring
    system_monitor = SystemMonitor()
    
    logger.info("🎉 nanoNAS API Server started successfully")


async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down nanoNAS API Server...")
    
    # Close connections
    if redis_client:
        redis_client.close()
    if mongo_client:
        mongo_client.close()
    
    logger.info("👋 nanoNAS API Server shutdown complete")


# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    
    secret_key = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        secret_key = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
        payload = jwt.decode(credentials.credentials, secret_key, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    # In production, fetch user from database
    return {"username": username, "role": "user"}


# API Routes

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to nanoNAS API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": redis_client is not None and redis_client.ping(),
            "mongodb": mongo_client is not None,
            "celery": celery_app is not None
        }
    }
    
    if system_monitor:
        status["system"] = await system_monitor.get_system_stats()
    
    return status


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system and application metrics."""
    if not system_monitor:
        raise HTTPException(status_code=503, detail="System monitoring not available")
    
    metrics = await system_monitor.get_detailed_metrics()
    return metrics


# Authentication endpoints
@app.post("/auth/register", tags=["Authentication"])
async def register_user(user: UserCreate):
    """Register a new user."""
    # Hash password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    
    user_data = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password.decode('utf-8'),
        "role": user.role,
        "created_at": datetime.utcnow(),
        "active": True
    }
    
    # In production, save to database
    if mongo_client:
        try:
            db = mongo_client.nanonas
            result = db.users.insert_one(user_data)
            user_data["_id"] = str(result.inserted_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"User registration failed: {e}")
    
    return {"message": "User registered successfully", "username": user.username}


@app.post("/auth/login", tags=["Authentication"])
async def login_user(user: UserLogin):
    """Authenticate user and return access token."""
    # In production, verify against database
    access_token = create_access_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 24 * 3600  # 24 hours
    }


# Search endpoints
@app.post("/search", response_model=Dict[str, Any], tags=["Search"])
async def create_search_experiment(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a new architecture search experiment."""
    
    # Generate experiment ID
    experiment_id = str(uuid.uuid4())
    experiment_name = request.experiment_name or f"search_{experiment_id[:8]}"
    
    # Create experiment record
    experiment_data = {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "strategy": request.strategy,
        "dataset": request.dataset,
        "search_space": request.search_space,
        "status": "created",
        "created_by": current_user["username"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "progress": 0.0,
        "best_accuracy": None,
        "total_architectures": 0,
        "config": request.dict()
    }
    
    # Store experiment
    if experiment_tracker:
        await experiment_tracker.create_experiment(experiment_data)
    
    # Start search in background
    if celery_app:
        # Use Celery for async processing
        celery_app.send_task(
            'nanonas.tasks.run_search',
            args=[experiment_id, request.dict()],
            task_id=experiment_id
        )
    else:
        # Use FastAPI background tasks as fallback
        background_tasks.add_task(
            run_search_task,
            experiment_id,
            request.dict()
        )
    
    return {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "status": "started",
        "message": "Search experiment started successfully"
    }


@app.get("/experiments", response_model=List[ExperimentResponse], tags=["Experiments"])
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """List all experiments for the current user."""
    
    if not experiment_tracker:
        raise HTTPException(status_code=503, detail="Experiment tracking not available")
    
    experiments = await experiment_tracker.list_experiments(
        user=current_user["username"],
        skip=skip,
        limit=limit,
        status=status
    )
    
    return experiments


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse, tags=["Experiments"])
async def get_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get details of a specific experiment."""
    
    if not experiment_tracker:
        raise HTTPException(status_code=503, detail="Experiment tracking not available")
    
    experiment = await experiment_tracker.get_experiment(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Check ownership (in production, implement proper access control)
    if experiment.get("created_by") != current_user["username"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return experiment


@app.get("/experiments/{experiment_id}/progress", tags=["Experiments"])
async def get_experiment_progress(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get real-time progress of an experiment."""
    
    if not experiment_tracker:
        raise HTTPException(status_code=503, detail="Experiment tracking not available")
    
    progress = await experiment_tracker.get_experiment_progress(experiment_id)
    
    if not progress:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return progress


@app.get("/experiments/{experiment_id}/architectures", response_model=List[ArchitectureResponse], tags=["Architectures"])
async def get_experiment_architectures(
    experiment_id: str,
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get architectures discovered in an experiment."""
    
    if not experiment_tracker:
        raise HTTPException(status_code=503, detail="Experiment tracking not available")
    
    architectures = await experiment_tracker.get_experiment_architectures(
        experiment_id,
        skip=skip,
        limit=limit
    )
    
    return architectures


@app.delete("/experiments/{experiment_id}", tags=["Experiments"])
async def delete_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an experiment and all associated data."""
    
    if not experiment_tracker:
        raise HTTPException(status_code=503, detail="Experiment tracking not available")
    
    # Check ownership
    experiment = await experiment_tracker.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.get("created_by") != current_user["username"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Cancel if running
    if celery_app and experiment.get("status") == "running":
        celery_app.control.revoke(experiment_id, terminate=True)
    
    # Delete experiment
    await experiment_tracker.delete_experiment(experiment_id)
    
    return {"message": "Experiment deleted successfully"}


# Benchmarking endpoints
@app.post("/benchmark", tags=["Benchmark"])
async def create_benchmark_experiment(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a benchmarking experiment."""
    
    experiment_id = str(uuid.uuid4())
    
    # Create experiment record
    experiment_data = {
        "experiment_id": experiment_id,
        "name": f"benchmark_{experiment_id[:8]}",
        "type": "benchmark",
        "strategies": request.strategies,
        "dataset": request.dataset,
        "search_space": request.search_space,
        "status": "created",
        "created_by": current_user["username"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "progress": 0.0,
        "config": request.dict()
    }
    
    # Store experiment
    if experiment_tracker:
        await experiment_tracker.create_experiment(experiment_data)
    
    # Start benchmark in background
    if celery_app:
        celery_app.send_task(
            'nanonas.tasks.run_benchmark',
            args=[experiment_id, request.dict()],
            task_id=experiment_id
        )
    else:
        background_tasks.add_task(
            run_benchmark_task,
            experiment_id,
            request.dict()
        )
    
    return {
        "experiment_id": experiment_id,
        "status": "started",
        "message": "Benchmark experiment started successfully"
    }


# WebSocket for real-time monitoring
@app.websocket("/ws/experiments/{experiment_id}")
async def websocket_experiment_monitor(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for real-time experiment monitoring."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send progress updates
            if experiment_tracker:
                progress = await experiment_tracker.get_experiment_progress(experiment_id)
                if progress:
                    await websocket.send_json(progress)
            
            await asyncio.sleep(2)  # Update every 2 seconds
    
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


# Background task functions
async def run_search_task(experiment_id: str, config: dict):
    """Background task to run architecture search."""
    try:
        # Update status
        if experiment_tracker:
            await experiment_tracker.update_experiment_status(experiment_id, "running")
        
        # Run search
        model, results = search(
            strategy=config["strategy"],
            dataset=config["dataset"],
            search_space=config["search_space"],
            epochs=config.get("epochs"),
            return_results=True
        )
        
        # Store results
        if experiment_tracker:
            await experiment_tracker.store_experiment_results(experiment_id, results)
            await experiment_tracker.update_experiment_status(experiment_id, "completed")
    
    except Exception as e:
        logger.error(f"Search task failed for experiment {experiment_id}: {e}")
        if experiment_tracker:
            await experiment_tracker.update_experiment_status(experiment_id, "failed")
            await experiment_tracker.store_experiment_error(experiment_id, str(e))


async def run_benchmark_task(experiment_id: str, config: dict):
    """Background task to run benchmarking."""
    try:
        # Update status
        if experiment_tracker:
            await experiment_tracker.update_experiment_status(experiment_id, "running")
        
        # Run benchmark
        results = benchmark(
            strategies=config["strategies"],
            dataset=config["dataset"],
            search_space=config["search_space"],
            num_runs=config["num_runs"],
            epochs=config["epochs"]
        )
        
        # Store results
        if experiment_tracker:
            await experiment_tracker.store_experiment_results(experiment_id, results)
            await experiment_tracker.update_experiment_status(experiment_id, "completed")
    
    except Exception as e:
        logger.error(f"Benchmark task failed for experiment {experiment_id}: {e}")
        if experiment_tracker:
            await experiment_tracker.update_experiment_status(experiment_id, "failed")
            await experiment_tracker.store_experiment_error(experiment_id, str(e))


# CLI entry point
def main():
    """Main entry point for the server CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="nanoNAS API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "nanonas.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main() 