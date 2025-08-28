from __future__ import annotations
import os, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from config import MODEL_ID, DEFAULT_SCHEMA
from schemas import CampaignRequest, CampaignResponse
from generator import generate_campaign_plan
from model_loader import load_llama

import uvicorn

app = FastAPI(title="Campaign Ideation API (Llama 3.1 8B)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    try:
        load_llama()
        return {"status": "ok", "model": MODEL_ID}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status":"error","detail":str(e)})

@app.get("/version")
def version():
    import transformers, torch, jsonschema
    return {
        "model": MODEL_ID,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "jsonschema": jsonschema.__version__,
    }

@app.get("/schema")
def schema():
    return DEFAULT_SCHEMA

@app.post("/campaign/generate", response_model=CampaignResponse)
def generate(req: CampaignRequest):
    brief = {
        "industry": req.industry,
        "audience": req.audience.dict(),
        "budget_thb": req.budget_thb,
        "objective": req.objective,
        "constraints": (req.constraints.dict() if req.constraints else {}),
        "language": req.language
    }
    try:
        plan, meta = generate_campaign_plan(brief, DEFAULT_SCHEMA)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return CampaignResponse(
        status="ok",
        plan=plan,
        model=MODEL_ID,
        elapsed_ms=meta.get("elapsed_ms", 0),
        warnings=meta.get("warnings"),
        brief_echo=req
    )

if __name__ == "__main__":
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, log_level="info")