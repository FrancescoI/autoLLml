from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import threading
import asyncio
import json

from backend.models import TrainingParams, TrainingStatusResponse, TrainingStatus, LogEntry
from backend.services.training_service import training_service

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start")
async def start_training(params: TrainingParams):
    if training_service.is_running:
        raise HTTPException(status_code=400, detail="Training is already running")
    
    training_service.reset()
    
    thread = threading.Thread(
        target=training_service.run,
        kwargs={
            "max_iterations": params.max_iterations,
            "mlflow_experiment_name": params.mlflow_experiment_name or "AutoLLml_Experiments",
            "mlflow_tracking_enabled": params.mlflow_tracking_enabled
        }
    )
    thread.daemon = True
    thread.start()
    
    return {"message": "Training started", "status": "running"}


@router.post("/stop")
async def stop_training():
    if not training_service.is_running:
        raise HTTPException(status_code=400, detail="No training is running")
    
    training_service.stop()
    return {"message": "Training stopped"}


@router.get("/status", response_model=TrainingStatusResponse)
async def get_status():
    history = training_service.get_history()
    completed = len([h for h in history if h.get("metric") is not None])
    best = max([h for h in history if h.get("metric") is not None], key=lambda x: x["metric"], default=None)
    
    return TrainingStatusResponse(
        status=TrainingStatus(training_service.status),
        current_iteration=training_service._current_iteration,
        max_iterations=training_service._max_iterations,
        iterations_completed=completed,
        best_metric=best["metric"] if best else None
    )


@router.get("/logs")
async def get_logs():
    logs = training_service.get_logs()
    return {"logs": logs, "count": len(logs)}


@router.get("/stream")
async def stream_logs():
    async def event_generator():
        last_count = 0
        while True:
            await asyncio.sleep(1)
            
            current_count = training_service.get_new_logs_count()
            
            if current_count > last_count or training_service.status != "running":
                logs = training_service.get_logs()
                
                data = {
                    "type": "update",
                    "status": training_service.status,
                    "current_iteration": training_service._current_iteration,
                    "logs": logs[-50:] if len(logs) > 50 else logs,
                    "history": training_service.get_history()
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                last_count = current_count
                
                if training_service.status not in ["running", "idle"]:
                    break
            
            if training_service.status not in ["running", "idle"]:
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history")
async def get_history():
    return {"history": training_service.get_history()}


@router.get("/strategy")
async def get_strategy():
    strategy = training_service.get_business_strategy()
    return {"strategy": strategy}
