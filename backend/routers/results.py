from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import re
from pathlib import Path

from backend.models import EvaluationReport, TrainingHistory
from backend.utils.file_manager import (
    get_evaluation_report,
    get_plots,
    get_dynamic_features_code,
    PLOTS_DIR,
    BASE_DIR
)
from backend.services.training_service import training_service

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/latest")
async def get_latest_report():
    report = get_evaluation_report()
    if not report:
        raise HTTPException(status_code=404, detail="No evaluation report found")
    return report


@router.get("/history")
async def get_training_history():
    history = training_service.get_history()
    best = max([h for h in history if h.get("metric") is not None], key=lambda x: x["metric"], default=None)
    
    return TrainingHistory(
        iterations=history,
        best_iteration=best["iteration"] if best else None,
        best_metric=best["metric"] if best else None
    )


@router.get("/plots")
async def get_plot_list():
    plots = get_plots()
    return {"plots": plots, "count": len(plots)}


@router.get("/plot/{plot_name}")
async def get_plot(plot_name: str):
    plot_path = PLOTS_DIR / plot_name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(
        str(plot_path),
        media_type="image/png",
        filename=plot_name
    )


@router.get("/features")
async def get_features():
    code = get_dynamic_features_code()
    if not code:
        return {"features": [], "code": ""}
    
    matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', code)
    TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target', 'target_col']
    features = [m for m in matches if m not in TARGET_CANDIDATES]
    
    return {
        "features": features,
        "count": len(features)
    }


@router.get("/code")
async def get_code():
    code = get_dynamic_features_code()
    return {"code": code}


@router.get("/correlations")
async def get_correlations():
    report = get_evaluation_report()
    if not report:
        raise HTTPException(status_code=404, detail="No evaluation report found")
    
    correlations = report.get("top_correlations_with_target", {})
    return {"correlations": correlations}


@router.get("/importance")
async def get_importance():
    report = get_evaluation_report()
    if not report:
        raise HTTPException(status_code=404, detail="No evaluation report found")
    
    importance = report.get("feature_importance", {})
    return {"importance": importance}
