"""
Reasoning / Citation Validation / Confidence / Abstention / final response models.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    point: str
    evidence_ids: List[str] = Field(default_factory=list)
    anchor: str = ""  # short verbatim phrase copied from the cited evidence (language-independent grounding check)


class ReasoningOutput(BaseModel):
    """Structured output of the LLM Reasoning Engine (stage 6)."""

    answer: str
    key_points: List[KeyPoint] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    should_abstain: bool = False
    provider: str = "unknown"


class ClaimValidation(BaseModel):
    claim: str
    supported: bool
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    strength: float = 0.0  # graded grounding strength in [0, 1]
    repaired: bool = False


class CitationValidationResult(BaseModel):
    claims: List[ClaimValidation] = Field(default_factory=list)
    citation_coverage: float = 0.0
    unsupported_claims: List[str] = Field(default_factory=list)
    citation_correctness: float = 0.0
    faithfulness: float = 0.0


class ConfidenceResult(BaseModel):
    score: float = 0.0
    level: str = "low"  # low | medium | high
    # None marks a signal that is not applicable (e.g. no product classification).
    factors: Dict[str, Optional[float]] = Field(default_factory=dict)


class AbstentionDecision(BaseModel):
    should_abstain: bool = False
    reason: Optional[str] = None
    missing_information: List[str] = Field(default_factory=list)
    recommended_next_step: Optional[str] = None


class QueryResponse(BaseModel):
    """POST /api/v1/query response body."""

    query_id: str
    status: str  # answered | abstained | error
    query: str
    language: str = "en"
    analysis: Dict[str, Any]
    classification: Dict[str, Any]
    routing: Dict[str, Any]
    decision_path: List[Dict[str, Any]] = Field(default_factory=list)
    graph_context: Dict[str, Any] = Field(default_factory=dict)
    answer: str
    key_points: List[str] = Field(default_factory=list)
    confidence: Dict[str, Any]
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    citation_coverage: float = 0.0
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    applicable_domains: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    abstained: bool = False
    warnings: List[str] = Field(default_factory=list)
    disclaimer: str
