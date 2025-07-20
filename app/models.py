"""
Pydantic models for ProfileScore API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class CandidateStatus(str, Enum):
    """Candidate status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    HIRED = "hired"
    REJECTED = "rejected"


class ScoringRequest(BaseModel):
    """Request model for profile scoring."""
    
    candidate_profile: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Candidate's professional profile/resume text"
    )
    
    job_description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Job description text to match against"
    )
    
    candidate_id: Optional[str] = Field(
        None,
        description="Unique identifier for the candidate"
    )
    
    job_id: Optional[str] = Field(
        None,
        description="Unique identifier for the job posting"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the scoring request"
    )
    
    @validator('candidate_profile', 'job_description')
    def validate_text_content(cls, v):
        """Validate that text contains meaningful content."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class CandidateData(BaseModel):
    """Model for storing candidate data in Pinecone."""
    
    candidate_id: str = Field(..., description="Unique candidate identifier")
    name: Optional[str] = Field(None, description="Candidate name")
    email: Optional[str] = Field(None, description="Candidate email")
    profile_text: str = Field(..., description="Full profile/resume text")
    skills: List[str] = Field(default_factory=list, description="Extracted skills")
    experience_years: Optional[int] = Field(None, description="Years of experience")
    current_role: Optional[str] = Field(None, description="Current job title")
    location: Optional[str] = Field(None, description="Candidate location")
    status: CandidateStatus = Field(default=CandidateStatus.ACTIVE, description="Candidate status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class JobData(BaseModel):
    """Model for storing job data in Pinecone."""
    
    job_id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    description: str = Field(..., description="Full job description")
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    experience_required: Optional[int] = Field(None, description="Required years of experience")
    location: Optional[str] = Field(None, description="Job location")
    salary_range: Optional[str] = Field(None, description="Salary range")
    is_active: bool = Field(default=True, description="Whether job posting is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ScoringResult(BaseModel):
    """Model for storing scoring results in Pinecone."""
    
    scoring_id: str = Field(..., description="Unique scoring identifier")
    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")
    score: int = Field(..., ge=0, le=100, description="Match score from 0 to 100")
    explanation: str = Field(..., description="Detailed explanation of the scoring")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    model_used: str = Field(..., description="Model used for scoring")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Scoring timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ScoringResponse(BaseModel):
    """Response model for profile scoring results."""
    
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Match score from 0 to 100"
    )
    
    explanation: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Detailed explanation of the scoring"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the scoring (0.0 to 1.0)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of when the scoring was performed"
    )
    
    model_used: str = Field(
        ...,
        description="Name/version of the model used for scoring"
    )
    
    scoring_id: Optional[str] = Field(
        None,
        description="Unique identifier for this scoring result"
    )
    
    candidate_id: Optional[str] = Field(
        None,
        description="Candidate identifier if provided"
    )
    
    job_id: Optional[str] = Field(
        None,
        description="Job identifier if provided"
    )


class CandidateSearchRequest(BaseModel):
    """Request model for searching candidates."""
    
    query: str = Field(..., description="Search query")
    job_id: Optional[str] = Field(None, description="Filter by job ID")
    min_score: Optional[int] = Field(None, ge=0, le=100, description="Minimum score filter")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    skills_filter: Optional[List[str]] = Field(None, description="Filter by required skills")
    location_filter: Optional[str] = Field(None, description="Filter by location")


class CandidateSearchResponse(BaseModel):
    """Response model for candidate search results."""
    
    candidates: List[Dict[str, Any]] = Field(..., description="List of matching candidates")
    total_results: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server timestamp"
    )
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    pinecone_connected: bool = Field(..., description="Whether Pinecone is connected")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when error occurred"
    ) 