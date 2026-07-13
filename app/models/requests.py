# app/models/requests.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class StoreMetadata(BaseModel):
    created_on: str = Field(..., description="Record creation timestamp (UTC)")
    image_path: str = Field(..., description="Path to the stored image")
    # Identity fields (optional, backward compatible). id_number is the
    # authoritative government identifier (PAN / Aadhaar-ref).
    customer_id: Optional[str] = Field(None, description="Bank customer / application id")
    full_name: Optional[str] = Field(None, description="Customer full name")
    id_type: Optional[str] = Field(None, description="Government ID type, e.g. PAN, AADHAAR")
    id_number: Optional[str] = Field(None, description="Government ID number")
    phone: Optional[str] = Field(None, description="Registered phone number")
    dob: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")


class SearchMetadata(BaseModel):
    created_on: str = Field(..., description="Record creation timestamp (UTC)")
    image_path: str = Field(..., description="Path to the stored image")


class CheckMetadata(BaseModel):
    """Identity of the applicant being screened at onboarding. All fields are
    optional so the demo stays flexible; id_number drives the same/different-
    identity fraud decision."""
    customer_id: Optional[str] = Field(None, description="Bank customer / application id")
    full_name: Optional[str] = Field(None, description="Applicant full name")
    id_type: Optional[str] = Field(None, description="Government ID type, e.g. PAN, AADHAAR")
    id_number: Optional[str] = Field(None, description="Government ID number")
    phone: Optional[str] = Field(None, description="Applicant phone number")
    dob: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    created_on: Optional[str] = Field(None, description="Record creation timestamp (UTC)")
    image_path: Optional[str] = Field(None, description="Path to the stored image")


class SearchRequest(BaseModel):
    threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")
    limit: Optional[int] = Field(1000, ge=1, le=1000, description="Maximum results")


class PurgeRequest(BaseModel):
    transaction_id: str = Field(..., description="ID of the transaction record to purge")
