# app/models/responses.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from app.models.requests import StoreMetadata, SearchMetadata


class BaseResponse(BaseModel):
    status: str = Field(..., description="Response status")


class StoreResponse(BaseResponse):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    message: str = Field(..., description="Success message")


class SearchResult(BaseModel):
    similarity_score: float = Field(..., description="Similarity score")
    metadata: StoreMetadata = Field(..., description="Stored customer metadata")


class SearchResponse(BaseResponse):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    total_matches: int = Field(..., description="Number of matching records found")
    metadata: SearchMetadata = Field(..., description="Query metadata from request")
    results: List[SearchResult] = Field(..., description="List of matching records")


class MatchDetail(BaseModel):
    transaction_id: str = Field(..., description="Matched gallery record id")
    similarity_score: float = Field(..., description="Face similarity (0-1, 1=identical)")
    identity_match: bool = Field(..., description="True if the matched record is the same identity")
    field_diffs: List[str] = Field(default_factory=list, description="Identity fields present in both but differing")
    identity: Dict[str, Optional[str]] = Field(default_factory=dict, description="Matched record identity fields")
    image_path: Optional[str] = Field(None, description="Path to the matched face image")


class CheckResponse(BaseResponse):
    transaction_id: str = Field(..., description="Applicant transaction id")
    verdict: str = Field(..., description="CLEAR | REVIEW | DUPLICATE_SAME_IDENTITY | FRAUD_ALERT_DIFFERENT_IDENTITY")
    risk_score: int = Field(..., description="Fraud risk score 0-100")
    reason_codes: List[str] = Field(default_factory=list, description="Explainability codes for the verdict")
    enrolled: bool = Field(..., description="Whether the applicant face was added to the gallery")
    query_identity: Dict[str, Optional[str]] = Field(default_factory=dict, description="Applicant identity fields")
    best_match: Optional[MatchDetail] = Field(None, description="Highest-similarity gallery match")
    matches: List[MatchDetail] = Field(default_factory=list, description="All candidate matches above t_candidate")
    total_matches: int = Field(0, description="Number of candidate matches")


class PurgeResponse(BaseResponse):
    transaction_id: str = Field(..., description="ID of the purged transaction record")
    message: str = Field(..., description="Success message")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
