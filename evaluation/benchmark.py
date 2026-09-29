"""
Benchmark Dataset for IP-SAKTI Sahayak Phase-1 Evaluation.
Contains gold-standard test queries, expected intents, domains, jurisdictions, expected document IDs, and key terms.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class BenchmarkQuery(BaseModel):
    query_id: str
    query: str
    category: str  # 'exact_keyword', 'semantic', 'mixed', 'jurisdiction', 'versioning', 'distractor_rejection'
    expected_domain: Optional[str] = None
    expected_jurisdiction: Optional[str] = None
    expected_doc_ids: List[str]
    expected_keywords: List[str]


BENCHMARK_DATASET: List[BenchmarkQuery] = [
    # 1. Exact Keyword Query
    BenchmarkQuery(
        query_id="Q-001",
        query="What is Section 3(p) under the Indian Patents Act for traditional knowledge?",
        category="exact_keyword",
        expected_domain="Patents",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-PAT-001", "SYN-PAT-003"],
        expected_keywords=["Section 3(p)", "aggregation", "traditional knowledge"]
    ),

    # 2. Semantic Query
    BenchmarkQuery(
        query_id="Q-002",
        query="Can I patent a traditional Ayurvedic formulation by proving unexpected synergistic effects?",
        category="semantic",
        expected_domain="Patents",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-PAT-001", "SYN-PAT-003"],
        expected_keywords=["synergistic", "Ashwagandha", "bio-availability"]
    ),

    # 3. Mixed Query
    BenchmarkQuery(
        query_id="Q-003",
        query="Can a new herbal formulation be patented in India and what NBA clearance evidence is required?",
        category="mixed",
        expected_domain="Patents",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-PAT-002", "SYN-ABS-001", "SYN-ABS-002"],
        expected_keywords=["Biological Diversity Act", "NBA", "Phytopharmaceutical"]
    ),

    # 4. Jurisdiction Query (International Route)
    BenchmarkQuery(
        query_id="Q-004",
        query="What is the international PCT patent application filing route for herbal inventions?",
        category="jurisdiction",
        expected_domain="Patents",
        expected_jurisdiction="International",
        expected_doc_ids=["SYN-INT-002"],
        expected_keywords=["PCT", "Receiving Office", "WIPO", "National Phase"]
    ),

    # 5. Access and Benefit Sharing (ABS) Query
    BenchmarkQuery(
        query_id="Q-005",
        query="What benefit sharing fee percentage is mandatory for foreign entities under the Biological Diversity Act?",
        category="mixed",
        expected_domain="Biodiversity/ABS",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-ABS-001", "SYN-ABS-002"],
        expected_keywords=["Access and Benefit Sharing", "ABS", "0.1%", "0.5%"]
    ),

    # 6. Drug Classification Query (Classical vs Proprietary)
    BenchmarkQuery(
        query_id="Q-006",
        query="What is the difference between classical Ayurvedic medicine and proprietary Ayurvedic medicine?",
        category="semantic",
        expected_domain="Drug Classification",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-DRG-001", "SYN-DRG-002"],
        expected_keywords=["First Schedule", "Section 3(a)", "Section 3(h)"]
    ),

    # 7. International Nagoya Protocol & WIPO GRATK Disclosure
    BenchmarkQuery(
        query_id="Q-007",
        query="What are the mandatory origin disclosure requirements for traditional knowledge under the WIPO GRATK treaty?",
        category="jurisdiction",
        expected_domain="Traditional Knowledge",
        expected_jurisdiction="International",
        expected_doc_ids=["SYN-INT-004"],
        expected_keywords=["WIPO GRATK", "country of origin", "mandatory patent disclosure"]
    ),

    # 8. Versioning Conflict Query (2024 vs 2026 Policy)
    BenchmarkQuery(
        query_id="Q-008",
        query="What is the current 2026 lead heavy metal limit for Ayurvedic herbal exports?",
        category="versioning",
        expected_domain="Drug Classification",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-VER-2026"],
        expected_keywords=["2026", "Lead (5.0 ppm)", "current"]
    ),

    # 9. Geographical Indications Query
    BenchmarkQuery(
        query_id="Q-009",
        query="Can an individual company claim exclusive ownership over a regional Ayurvedic Geographical Indication?",
        category="semantic",
        expected_domain="Geographical Indications",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-GI-001", "SYN-GI-002"],
        expected_keywords=["Geographical Indication", "community ownership", "association"]
    ),

    # 10. Distractor Rejection Query
    BenchmarkQuery(
        query_id="Q-010",
        query="What are the patent guidelines for traditional herbal formulations?",
        category="distractor_rejection",
        expected_domain="Patents",
        expected_jurisdiction="India",
        expected_doc_ids=["SYN-PAT-001", "SYN-PAT-004"],
        expected_keywords=["traditional knowledge", "novelty"]
    )
]
