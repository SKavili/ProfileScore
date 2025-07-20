"""
Main FastAPI application for ProfileScore.
"""

import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import (
    ScoringRequest, 
    ScoringResponse, 
    HealthResponse, 
    ErrorResponse,
    CandidateData,
    JobData,
    CandidateSearchRequest,
    CandidateSearchResponse
)
from app.services.llama_service import llama_service
from app.services.pinecone_service import pinecone_service
from app.services.langchain_service import langchain_service
from app.utils.logger import get_logger
from app import __version__


# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting ProfileScore application...")
    
    # Load the LLaMA 2 model
    logger.info("Loading LLaMA 2 model...")
    model_loaded = llama_service.load_model()
    
    if not model_loaded:
        logger.error("Failed to load LLaMA 2 model. Application may not function properly.")
    else:
        logger.info("LLaMA 2 model loaded successfully")
    
    # Initialize LangChain service
    logger.info("Initializing LangChain service...")
    langchain_initialized = langchain_service.initialize()
    
    if not langchain_initialized:
        logger.warning("Failed to initialize LangChain service. Enhanced features may not be available.")
    else:
        logger.info("LangChain service initialized successfully")
    
    # Connect to Pinecone
    logger.info("Connecting to Pinecone...")
    pinecone_connected = pinecone_service.connect()
    
    if not pinecone_connected:
        logger.warning("Failed to connect to Pinecone. Vector storage features may not be available.")
    else:
        logger.info("Pinecone connected successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ProfileScore application...")
    llama_service.unload_model()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="ProfileScore API",
    description="LLaMA 2-powered job profile matching and scoring service with Pinecone vector storage and LangChain integration",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ProfileScore API - LLaMA 2 Job Profile Matching with Pinecone & LangChain",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "features": [
            "LLaMA 2-powered scoring",
            "Pinecone vector storage",
            "LangChain enhanced processing",
            "Candidate management",
            "Job management",
            "Vector similarity search"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if llama_service.is_loaded else "degraded",
        version=__version__,
        model_loaded=llama_service.is_loaded,
        pinecone_connected=pinecone_service.is_connected
    )


@app.post("/score", response_model=ScoringResponse)
async def score_profile(request: ScoringRequest):
    """
    Score a candidate profile against a job description.
    
    Args:
        request: ScoringRequest containing candidate profile and job description
        
    Returns:
        ScoringResponse with score, explanation, and metadata
        
    Raises:
        HTTPException: If scoring fails or model is not available
    """
    try:
        logger.info("Received profile scoring request")
        
        if not llama_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLaMA 2 model is not loaded. Please try again later."
            )
        
        # Generate scoring ID
        scoring_id = f"score_{uuid.uuid4().hex[:8]}"
        
        # Perform scoring
        result = llama_service.score_profile(
            candidate_profile=request.candidate_profile,
            job_description=request.job_description
        )
        
        # Store scoring result in Pinecone if connected
        if pinecone_service.is_connected and request.candidate_id and request.job_id:
            scoring_result = ScoringResult(
                scoring_id=scoring_id,
                candidate_id=request.candidate_id,
                job_id=request.job_id,
                score=result.score,
                explanation=result.explanation,
                confidence=result.confidence,
                model_used=result.model_used,
                metadata=request.metadata or {}
            )
            pinecone_service.store_scoring_result(scoring_result)
        
        # Add IDs to response
        result.scoring_id = scoring_id
        result.candidate_id = request.candidate_id
        result.job_id = request.job_id
        
        logger.info(f"Profile scoring completed successfully. Score: {result.score}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile scoring failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score profile: {str(e)}"
        )


@app.post("/score/enhanced", response_model=dict)
async def enhanced_score_profile(request: ScoringRequest):
    """
    Enhanced scoring using LangChain with detailed analysis.
    
    Args:
        request: ScoringRequest containing candidate profile and job description
        
    Returns:
        Enhanced scoring results with detailed analysis
    """
    try:
        logger.info("Received enhanced profile scoring request")
        
        if not langchain_service.is_initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LangChain service is not initialized. Please try again later."
            )
        
        # Perform enhanced scoring
        result = langchain_service.enhanced_scoring(
            candidate_profile=request.candidate_profile,
            job_description=request.job_description
        )
        
        # Store in Pinecone if connected and IDs provided
        if pinecone_service.is_connected and request.candidate_id and request.job_id:
            scoring_id = f"score_{uuid.uuid4().hex[:8]}"
            scoring_result = ScoringResult(
                scoring_id=scoring_id,
                candidate_id=request.candidate_id,
                job_id=request.job_id,
                score=result["enhanced_score"],
                explanation=result["explanation"],
                confidence=result["confidence"],
                model_used=result["model_used"],
                metadata={
                    "enhanced_scoring": True,
                    "base_score": result["base_score"],
                    "skill_match_score": result["skill_match_score"],
                    "experience_match_score": result["experience_match_score"],
                    **(request.metadata or {})
                }
            )
            pinecone_service.store_scoring_result(scoring_result)
            result["scoring_id"] = scoring_id
        
        logger.info(f"Enhanced scoring completed. Score: {result['enhanced_score']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced scoring failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform enhanced scoring: {str(e)}"
        )


@app.post("/candidates", response_model=dict)
async def create_candidate(candidate_data: CandidateData):
    """
    Create and store a new candidate in Pinecone.
    
    Args:
        candidate_data: Candidate data to store
        
    Returns:
        Success response with candidate ID
    """
    try:
        logger.info(f"Creating candidate: {candidate_data.candidate_id}")
        
        if not pinecone_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone is not connected. Please try again later."
            )
        
        # Extract skills using LangChain if available
        if langchain_service.is_initialized:
            candidate_data.skills = langchain_service.extract_skills(candidate_data.profile_text)
        
        # Store candidate
        success = pinecone_service.store_candidate(candidate_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store candidate"
            )
        
        return {
            "message": "Candidate created successfully",
            "candidate_id": candidate_data.candidate_id,
            "skills_extracted": len(candidate_data.skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create candidate: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create candidate: {str(e)}"
        )


@app.get("/candidates/{candidate_id}", response_model=dict)
async def get_candidate(candidate_id: str):
    """
    Retrieve a candidate by ID.
    
    Args:
        candidate_id: Candidate ID to retrieve
        
    Returns:
        Candidate data
    """
    try:
        if not pinecone_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone is not connected. Please try again later."
            )
        
        candidate = pinecone_service.get_candidate(candidate_id)
        
        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Candidate {candidate_id} not found"
            )
        
        return candidate
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get candidate {candidate_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get candidate: {str(e)}"
        )


@app.post("/candidates/search", response_model=CandidateSearchResponse)
async def search_candidates(request: CandidateSearchRequest):
    """
    Search for candidates using vector similarity.
    
    Args:
        request: Search request with query and filters
        
    Returns:
        List of matching candidates
    """
    try:
        if not pinecone_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone is not connected. Please try again later."
            )
        
        candidates = pinecone_service.search_candidates(
            query=request.query,
            job_id=request.job_id,
            min_score=request.min_score,
            max_results=request.max_results,
            skills_filter=request.skills_filter,
            location_filter=request.location_filter
        )
        
        return CandidateSearchResponse(
            candidates=candidates,
            total_results=len(candidates),
            query=request.query,
            search_metadata={
                "filters_applied": {
                    "job_id": request.job_id,
                    "min_score": request.min_score,
                    "skills_filter": request.skills_filter,
                    "location_filter": request.location_filter
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search candidates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search candidates: {str(e)}"
        )


@app.post("/jobs", response_model=dict)
async def create_job(job_data: JobData):
    """
    Create and store a new job in Pinecone.
    
    Args:
        job_data: Job data to store
        
    Returns:
        Success response with job ID
    """
    try:
        logger.info(f"Creating job: {job_data.job_id}")
        
        if not pinecone_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone is not connected. Please try again later."
            )
        
        # Extract requirements using LangChain if available
        if langchain_service.is_initialized:
            requirements = langchain_service.extract_job_requirements(job_data.description)
            job_data.required_skills = requirements.get("required_skills", [])
            job_data.preferred_skills = requirements.get("preferred_skills", [])
            job_data.experience_required = requirements.get("experience_required")
        
        # Store job
        success = pinecone_service.store_job(job_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store job"
            )
        
        return {
            "message": "Job created successfully",
            "job_id": job_data.job_id,
            "required_skills_extracted": len(job_data.required_skills),
            "preferred_skills_extracted": len(job_data.preferred_skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@app.get("/jobs/{job_id}", response_model=dict)
async def get_job(job_id: str):
    """
    Retrieve a job by ID.
    
    Args:
        job_id: Job ID to retrieve
        
    Returns:
        Job data
    """
    try:
        if not pinecone_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone is not connected. Please try again later."
            )
        
        job = pinecone_service.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return job
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job: {str(e)}"
        )


@app.get("/scoring/history", response_model=dict)
async def get_scoring_history(candidate_id: str = None, job_id: str = None, limit: int = 50):
    """
    Retrieve scoring history.
    
    Args:
        candidate_id: Filter by candidate ID
        job_id: Filter by job ID
        limit: Maximum number of results
        
    Returns:
        List of scoring results
    """
    try:
        if not pinecone_service.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone is not connected. Please try again later."
            )
        
        results = pinecone_service.get_scoring_history(
            candidate_id=candidate_id,
            job_id=job_id,
            limit=limit
        )
        
        return {
            "scoring_results": results,
            "total_results": len(results),
            "filters": {
                "candidate_id": candidate_id,
                "job_id": job_id,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scoring history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scoring history: {str(e)}"
        )


@app.post("/batch/score", response_model=dict)
async def batch_score_profiles(request: dict):
    """
    Score multiple candidates against a job description in batch.
    
    Args:
        request: Dictionary with candidates list and job description
        
    Returns:
        Batch scoring results
    """
    try:
        candidates = request.get("candidates", [])
        job_description = request.get("job_description", "")
        
        if not candidates or not job_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="candidates and job_description are required"
            )
        
        if not langchain_service.is_initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LangChain service is not initialized. Please try again later."
            )
        
        # Process batch
        results = langchain_service.process_batch(candidates, job_description)
        
        return {
            "results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch scoring failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform batch scoring: {str(e)}"
        )


@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded LLaMA 2 model."""
    try:
        model_info = llama_service.get_model_info()
        
        # Add LangChain info
        if langchain_service.is_initialized:
            model_info["langchain_initialized"] = True
        else:
            model_info["langchain_initialized"] = False
        
        # Add Pinecone info
        if pinecone_service.is_connected:
            model_info["pinecone_connected"] = True
            model_info["pinecone_stats"] = pinecone_service.get_index_stats()
        else:
            model_info["pinecone_connected"] = False
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model():
    """Reload the LLaMA 2 model."""
    try:
        logger.info("Reloading LLaMA 2 model...")
        
        # Unload current model
        llama_service.unload_model()
        
        # Load model again
        success = llama_service.load_model()
        
        if success:
            logger.info("Model reloaded successfully")
            return {"message": "Model reloaded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    ) 