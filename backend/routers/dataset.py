from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional

from backend.models import DatasetInfo
from backend.utils.file_manager import (
    ensure_directories,
    get_dataset_info,
    get_dataset_preview,
    DATA_DIR,
    DATASET_PATH
)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        ensure_directories()
        content = await file.read()
        DATASET_PATH.write_bytes(content)
        
        info = get_dataset_info()
        if not info:
            raise HTTPException(status_code=500, detail="Failed to read uploaded dataset")
        
        return {
            "message": "Dataset uploaded successfully",
            "dataset": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", response_model=DatasetInfo)
async def get_info():
    info = get_dataset_info()
    if not info:
        raise HTTPException(status_code=404, detail="No dataset found")
    return info


@router.get("/preview")
async def get_preview(rows: int = 10):
    preview = get_dataset_preview(rows)
    if preview is None:
        raise HTTPException(status_code=404, detail="No dataset found")
    return {"preview": preview, "rows": rows}


@router.delete("/")
async def delete_dataset():
    try:
        if DATASET_PATH.exists():
            DATASET_PATH.unlink()
        return {"message": "Dataset deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
