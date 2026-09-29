"""
Query Intake & Query Analysis models for the IP-SAKTI Sahayak Intelligence Layer.
"""
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    PATENTABILITY = "patentability"
    TRADEMARK = "trademark"
    COPYRIGHT = "copyright"
    DESIGN = "design"
    TRADE_SECRET = "trade_secret"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"
    BIODIVERSITY_ACCESS = "biodiversity_access"
    ABS = "abs"
    REGULATORY_CLASSIFICATION = "regulatory_classification"
    PRODUCT_COMPLIANCE = "product_compliance"
    LABELING = "labeling"
    IMPORT_EXPORT = "import_export"
    LICENSING = "licensing"
    PRIOR_ART = "prior_art"
    GENERAL_INFORMATION = "general_information"
    UNKNOWN = "unknown"


class IPDomain(str, Enum):
    PATENT = "patent"
    TRADEMARK = "trademark"
    COPYRIGHT = "copyright"
    DESIGN = "design"
    TRADE_SECRET = "trade_secret"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"


class RegulatoryDomain(str, Enum):
    AYURVEDA = "ayurveda"
    DRUG = "drug"
    FOOD = "food"
    COSMETIC = "cosmetic"
    BIODIVERSITY = "biodiversity"
    ABS = "abs"
    IMPORT_EXPORT = "import_export"


# Jurisdiction-sensitive intents MUST have an unambiguous jurisdiction before
# the system is willing to answer with confidence (see services/abstention.py).
JURISDICTION_SENSITIVE_INTENTS = {
    QueryIntent.PATENTABILITY,
    QueryIntent.TRADEMARK,
    QueryIntent.COPYRIGHT,
    QueryIntent.DESIGN,
    QueryIntent.TRADE_SECRET,
    QueryIntent.TRADITIONAL_KNOWLEDGE,
    QueryIntent.BIODIVERSITY_ACCESS,
    QueryIntent.ABS,
    QueryIntent.REGULATORY_CLASSIFICATION,
    QueryIntent.PRODUCT_COMPLIANCE,
    QueryIntent.LABELING,
    QueryIntent.IMPORT_EXPORT,
    QueryIntent.LICENSING,
}

PRODUCT_SENSITIVE_INTENTS = {
    QueryIntent.REGULATORY_CLASSIFICATION,
    QueryIntent.PRODUCT_COMPLIANCE,
    QueryIntent.LABELING,
}


class QueryIntakeRequest(BaseModel):
    """POST /api/v1/query request body."""

    query: str = Field(..., min_length=1, json_schema_extra={
        "example": "Can I patent a traditional Ayurvedic formulation containing Ashwagandha under Section 3(p)?"
    })
    session_id: Optional[str] = Field(default=None, description="Client-supplied session id; generated if absent.")
    language: str = Field(default="auto", description="'auto' to detect, or an ISO language code.")
    jurisdiction: str = Field(default="auto", description="'auto' to infer, or an explicit jurisdiction such as 'India'.")
    top_k: int = Field(default=5, ge=1, le=20)
    product_description: Optional[str] = Field(
        default=None,
        description="Optional free-text product/ingredient/claims description used for product classification."
    )


class NormalizedQuery(BaseModel):
    """Output of Query Intake (stage 3.1)."""

    query_id: str
    session_id: str
    original_query: str
    normalized_query: str
    language: str
    requested_jurisdiction: str
    top_k: int
    product_description: Optional[str] = None


class QueryAnalysis(BaseModel):
    """Output of the Query Analyzer (stage 3.2)."""

    intent: QueryIntent = QueryIntent.UNKNOWN
    jurisdiction: str = Field(default="unknown", description="Resolved jurisdiction, or 'ambiguous'/'unknown'.")
    jurisdiction_candidates: List[str] = Field(default_factory=list)
    ip_domains: List[IPDomain] = Field(default_factory=list)
    regulatory_domains: List[RegulatoryDomain] = Field(default_factory=list)
    ayurveda_relevance: bool = False
    traditional_knowledge_relevance: bool = False
    biological_resource_relevance: bool = False
    destination_market: Optional[str] = None
    missing_information: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
