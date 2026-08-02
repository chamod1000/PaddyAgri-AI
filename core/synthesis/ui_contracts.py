"""
UIContracts Module - Version 3.0 Architecture
Defines first-class UI component contracts for zero-LLM rendering pathways.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class WeatherCardContract(BaseModel):
    location: str
    temperature_c: float
    humidity_pct: int
    rainfall_mm: float
    fungal_risk_alert: str
    timing_advice: str


class FertilizerTableContract(BaseModel):
    season: str
    acreage: float
    urea_kg: float
    tsp_kg: float
    mop_kg: float
    application_schedule: List[str]


class MarketGridContract(BaseModel):
    district: str
    samba_price_lkr: float
    nadu_price_lkr: float
    keeri_samba_price_lkr: float
    updated_date: str
