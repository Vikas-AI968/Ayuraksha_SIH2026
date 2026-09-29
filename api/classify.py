from fastapi import APIRouter

from models.classification import ProductClassificationRequest, ProductClassificationResult
from services.classifier import ProductClassifier

router = APIRouter(prefix="/api/v1", tags=["Classification"])

classifier = ProductClassifier()


@router.post("/classify", response_model=ProductClassificationResult)
def classify_product(payload: ProductClassificationRequest):
    """
    Classifies an Ayurveda-adjacent product into one of:
    classical_medicine | proprietary_medicine | new_drug | phytopharmaceutical |
    ayurveda_aahar | cosmetic | unknown.
    """
    return classifier.classify(payload)
