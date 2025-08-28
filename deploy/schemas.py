from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Audience(BaseModel):
    geo: str = Field(..., example="TH")
    age: str = Field(..., example="18-24")

class Constraints(BaseModel):
    brand_tone: Optional[str] = Field(None, example="playful")
    mandatory_channels: Optional[List[str]] = Field(default_factory=list)
    banned_channels: Optional[List[str]] = Field(default_factory=list)

class CampaignRequest(BaseModel):
    industry: str = Field(..., example="FMCG snacks")
    audience: Audience
    budget_thb: float = Field(..., gt=50000)
    objective: str = Field(..., example="awareness")
    constraints: Optional[Constraints] = None
    language: Optional[str] = Field(None, description="Optional hint for output copy language, e.g., 'TH' or 'EN'")

class CampaignResponse(BaseModel):
    status: str
    plan: Dict[str, Any]
    model: str
    elapsed_ms: int
    warnings: Optional[List[str]] = None
    brief_echo: CampaignRequest