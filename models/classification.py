"""
Product Classification models for the IP-SAKTI Sahayak Intelligence Layer.
"""
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class ProductClass(str, Enum):
    CLASSICAL_MEDICINE = "classical_medicine"
    PROPRIETARY_MEDICINE = "proprietary_medicine"
    NEW_DRUG = "new_drug"
    PHYTOPHARMACEUTICAL = "phytopharmaceutical"
    AYURVEDA_AAHAR = "ayurveda_aahar"
    COSMETIC = "cosmetic"
    UNKNOWN = "unknown"


class ProductClassificationRequest(BaseModel):
    """POST /api/v1/classify request body."""

    product_description: str = Field(..., min_length=1, json_schema_extra={
        "example": "A proprietary herbal capsule containing Ashwagandha and Brahmi extract, sold under a brand name, marketed for stress relief."
    })
    ingredients: Optional[List[str]] = Field(default=None)
    intended_use: Optional[str] = Field(default=None)
    dosage_form: Optional[str] = Field(default=None, description="e.g. tablet, capsule, churna, cream, syrup")
    claims: Optional[List[str]] = Field(default=None, description="Marketing / therapeutic claims made about the product")


class ProductClassificationResult(BaseModel):
    classification: ProductClass = ProductClass.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    applicable_domains: List[str] = Field(default_factory=list)
