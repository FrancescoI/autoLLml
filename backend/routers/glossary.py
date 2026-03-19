from fastapi import APIRouter, HTTPException

from backend.models import GlossaryContent
from backend.utils.file_manager import get_glossary, save_glossary

router = APIRouter(prefix="/api/glossary", tags=["glossary"])


@router.get("/")
async def get_glossary_content():
    content = get_glossary()
    return {"content": content}


@router.post("/save")
async def save_glossary_content(data: GlossaryContent):
    success = save_glossary(data.content)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save glossary")
    return {"message": "Glossary saved successfully"}
