"""
backend/routers/images.py
Image generation endpoints.

POST /api/v1/images/generate  → auto-decide visualization type via ImageAgent
POST /api/v1/images/dalle     → force DALL-E 3 image
POST /api/v1/images/chart     → force matplotlib chart
"""

import base64
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from dependencies import get_current_user
from model_manager import get_llm

router = APIRouter()


class GenerateRequest(BaseModel):
    concept: str
    model: str = "Llama 3.3 70B"


class DalleRequest(BaseModel):
    prompt: str


class ChartRequest(BaseModel):
    chart_type: str = "bar"        # bar, pie, line
    title: str = ""
    labels: list[str] = []
    values: list[float] = []
    xlabel: str = ""
    ylabel: str = ""


class FlowchartRequest(BaseModel):
    title: str
    steps: list[str]


@router.post("/generate")
async def generate_image(
    body: GenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Auto-decide visualization type using ImageAgent, then generate.
    Returns base64-encoded PNG + metadata.
    """
    from agents.image_agent import ImageAgent
    from services.image_service import generate_visualization

    if not body.concept.strip():
        raise HTTPException(status_code=400, detail="Concept cannot be empty")

    llm = get_llm(body.model)
    agent = ImageAgent(llm)
    decision = agent.decide(body.concept)

    img_bytes = generate_visualization(decision)

    if not img_bytes:
        raise HTTPException(
            status_code=500,
            detail="Image generation failed. Check OPENAI_API_KEY or matplotlib."
        )

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "type": decision.get("type", "dalle"),
        "decision": decision,
    }


@router.post("/dalle")
async def force_dalle(
    body: DalleRequest,
    user_id: str = Depends(get_current_user)
):
    """Force DALL-E 3 image generation."""
    from services.image_service import generate_dalle_image

    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    img_bytes = generate_dalle_image(body.prompt)
    if not img_bytes:
        raise HTTPException(status_code=500, detail="DALL-E generation failed")

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "type": "dalle",
    }


@router.post("/chart")
async def force_chart(
    body: ChartRequest,
    user_id: str = Depends(get_current_user)
):
    """Force matplotlib chart generation."""
    from services.image_service import generate_chart

    img_bytes = generate_chart(
        chart_type=body.chart_type,
        title=body.title,
        labels=body.labels,
        values=body.values,
        xlabel=body.xlabel,
        ylabel=body.ylabel,
    )
    if not img_bytes:
        raise HTTPException(status_code=500, detail="Chart generation failed")

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "type": "chart",
    }


@router.post("/flowchart")
async def force_flowchart(
    body: FlowchartRequest,
    user_id: str = Depends(get_current_user)
):
    """Force matplotlib flowchart generation."""
    from services.image_service import generate_flowchart

    img_bytes = generate_flowchart(title=body.title, steps=body.steps)
    if not img_bytes:
        raise HTTPException(status_code=500, detail="Flowchart generation failed")

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "type": "flowchart",
    }
