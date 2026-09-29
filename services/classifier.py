"""
Product Classification service (stage 3.3).

Deterministic, keyword/rule-based classifier over ingredients, formulation,
intended use, dosage form, and claims. Returns "unknown" rather than
guessing when there isn't enough signal -- see RULE: "Do NOT hallucinate a
classification."
"""
from __future__ import annotations

from typing import List, Optional

from models.classification import (
    ProductClass,
    ProductClassificationRequest,
    ProductClassificationResult,
)

_COSMETIC_KEYWORDS = ["cosmetic", "skin cream", "topical", "external use only", "lotion", "moisturi", "soap", "shampoo"]
_FOOD_KEYWORDS = ["aahara", "aahar", "food", "dietary", "nutraceutical", "supplement", "beverage", "snack"]
_PHYTOPHARMA_KEYWORDS = [
    "phytopharmaceutical", "standardized extract", "standardised extract", "purified fraction",
    "botanical drug", "isolated phytoconstituent",
]
_NEW_DRUG_KEYWORDS = [
    "new drug", "novel formulation", "clinical trial", "new combination", "not described in", "modern research",
    "not found in classical texts",
]
_PROPRIETARY_KEYWORDS = [
    "proprietary", "brand name", "branded", "patented formulation", "trademarked", "own formulation",
]
_CLASSICAL_KEYWORDS = [
    "classical formulation", "shastriya", "charaka samhita", "sushruta samhita", "ashtanga hridaya",
    "as per classical texts", "traditional recipe", "authoritative ayurvedic text",
]

_DISEASE_CLAIM_KEYWORDS = ["cure", "treat", "therapeutic", "disease", "cures", "treats", "medicinal claim"]


class ProductClassifier:
    def classify(self, request: ProductClassificationRequest) -> ProductClassificationResult:
        text = " ".join(filter(None, [
            request.product_description,
            " ".join(request.ingredients or []),
            request.intended_use or "",
            request.dosage_form or "",
            " ".join(request.claims or []),
        ])).lower()

        reasons: List[str] = []
        missing_information: List[str] = []
        applicable_domains: List[str] = ["Drug Classification"]

        scores = {
            ProductClass.CLASSICAL_MEDICINE: sum(1 for k in _CLASSICAL_KEYWORDS if k in text),
            ProductClass.PROPRIETARY_MEDICINE: sum(1 for k in _PROPRIETARY_KEYWORDS if k in text),
            ProductClass.NEW_DRUG: sum(1 for k in _NEW_DRUG_KEYWORDS if k in text),
            ProductClass.PHYTOPHARMACEUTICAL: sum(1 for k in _PHYTOPHARMA_KEYWORDS if k in text),
            ProductClass.AYURVEDA_AAHAR: sum(1 for k in _FOOD_KEYWORDS if k in text),
            ProductClass.COSMETIC: sum(1 for k in _COSMETIC_KEYWORDS if k in text),
        }

        has_disease_claim = any(k in text for k in _DISEASE_CLAIM_KEYWORDS)
        # A cosmetic cannot carry a disease/therapeutic claim; downweight if it does.
        if scores[ProductClass.COSMETIC] and has_disease_claim:
            scores[ProductClass.COSMETIC] = max(0, scores[ProductClass.COSMETIC] - 1)
            reasons.append("Therapeutic/disease claims present alongside cosmetic language -- cosmetic classification downweighted.")

        # Food classification should not coexist with strong disease claims either.
        if scores[ProductClass.AYURVEDA_AAHAR] and has_disease_claim:
            scores[ProductClass.AYURVEDA_AAHAR] = max(0, scores[ProductClass.AYURVEDA_AAHAR] - 1)
            reasons.append("Therapeutic/disease claims present alongside food language -- Ayurveda Aahar classification downweighted.")

        best_class, best_score = max(scores.items(), key=lambda kv: kv[1])

        if best_score == 0:
            missing_information.append(
                "Insufficient information to classify the product -- please provide ingredients, "
                "dosage form, intended use, and any label/marketing claims."
            )
            return ProductClassificationResult(
                classification=ProductClass.UNKNOWN,
                confidence=0.0,
                reasons=["No classification signal found in the supplied description/ingredients/claims."],
                missing_information=missing_information,
                applicable_domains=applicable_domains,
            )

        # Check for a tie among the top score -- ties are genuinely ambiguous.
        tied = [c for c, s in scores.items() if s == best_score]
        if len(tied) > 1:
            missing_information.append(
                f"Signals point to multiple possible classifications ({', '.join(c.value for c in tied)}); "
                "more specific product information is needed to disambiguate."
            )
            return ProductClassificationResult(
                classification=ProductClass.UNKNOWN,
                confidence=0.3,
                reasons=[f"Ambiguous between: {', '.join(c.value for c in tied)}."],
                missing_information=missing_information,
                applicable_domains=applicable_domains,
            )

        total_signal = sum(scores.values())
        confidence = round(min(0.95, 0.4 + 0.5 * (best_score / max(1, total_signal))), 2)

        reasons.append(f"Matched {best_score} keyword signal(s) most consistent with '{best_class.value}'.")
        if not request.ingredients:
            missing_information.append("Ingredient list was not provided; classification confidence may improve with it.")
        if not request.dosage_form:
            missing_information.append("Dosage form was not provided; classification confidence may improve with it.")

        if best_class in (ProductClass.CLASSICAL_MEDICINE, ProductClass.PROPRIETARY_MEDICINE,
                           ProductClass.NEW_DRUG, ProductClass.PHYTOPHARMACEUTICAL):
            applicable_domains.append("Traditional Knowledge")

        return ProductClassificationResult(
            classification=best_class,
            confidence=confidence,
            reasons=reasons,
            missing_information=missing_information,
            applicable_domains=applicable_domains,
        )
