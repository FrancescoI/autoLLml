from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routers import dataset_router, glossary_router, training_router, results_router

app = FastAPI(
    title="AutoML API",
    description="Backend API for AutoML Dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset_router)
app.include_router(glossary_router)
app.include_router(training_router)
app.include_router(results_router)


@app.get("/")
async def root():
    return FileResponse(str(Path(__file__).parent.parent / "frontend" / "index.html"))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
