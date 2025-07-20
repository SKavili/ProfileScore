"""
Pinecone service for vector storage and retrieval.
"""

import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np

from app.utils.logger import get_logger
from app.models import CandidateData, JobData, ScoringResult


class PineconeService:
    """Service class for Pinecone vector database operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.index = None
        self.embedding_model = None
        self.is_connected = False
        
        # Configuration
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "profilescores")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Vector dimensions for the embedding model
        self.vector_dimension = 384  # for all-MiniLM-L6-v2
        
    def connect(self) -> bool:
        """
        Connect to Pinecone and initialize the index.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if not self.api_key or not self.environment:
                raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT are required")
            
            self.logger.info("Initializing Pinecone connection...")
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists, create if not
            if self.index_name not in pinecone.list_indexes():
                self.logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.vector_dimension,
                    metric="cosine"
                )
            
            # Connect to the index
            self.index = pinecone.Index(self.index_name)
            
            # Load embedding model
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            self.is_connected = True
            self.logger.info("Pinecone connection established successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Pinecone: {str(e)}")
            self.is_connected = False
            return False
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")
        
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def store_candidate(self, candidate_data: CandidateData) -> bool:
        """
        Store candidate data in Pinecone.
        
        Args:
            candidate_data: Candidate data to store
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            # Generate embedding for profile text
            embedding = self._generate_embedding(candidate_data.profile_text)
            
            # Prepare metadata
            metadata = {
                "type": "candidate",
                "candidate_id": candidate_data.candidate_id,
                "name": candidate_data.name,
                "email": candidate_data.email,
                "skills": json.dumps(candidate_data.skills),
                "experience_years": candidate_data.experience_years,
                "current_role": candidate_data.current_role,
                "location": candidate_data.location,
                "status": candidate_data.status.value,
                "created_at": candidate_data.created_at.isoformat(),
                "updated_at": candidate_data.updated_at.isoformat(),
                "profile_text": candidate_data.profile_text[:1000],  # Truncate for metadata
                **candidate_data.metadata
            }
            
            # Store in Pinecone
            vector_id = f"candidate_{candidate_data.candidate_id}"
            self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            
            self.logger.info(f"Stored candidate: {candidate_data.candidate_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store candidate: {str(e)}")
            return False
    
    def store_job(self, job_data: JobData) -> bool:
        """
        Store job data in Pinecone.
        
        Args:
            job_data: Job data to store
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            # Generate embedding for job description
            embedding = self._generate_embedding(job_data.description)
            
            # Prepare metadata
            metadata = {
                "type": "job",
                "job_id": job_data.job_id,
                "title": job_data.title,
                "company": job_data.company,
                "required_skills": json.dumps(job_data.required_skills),
                "preferred_skills": json.dumps(job_data.preferred_skills),
                "experience_required": job_data.experience_required,
                "location": job_data.location,
                "salary_range": job_data.salary_range,
                "is_active": job_data.is_active,
                "created_at": job_data.created_at.isoformat(),
                "updated_at": job_data.updated_at.isoformat(),
                "description": job_data.description[:1000],  # Truncate for metadata
                **job_data.metadata
            }
            
            # Store in Pinecone
            vector_id = f"job_{job_data.job_id}"
            self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            
            self.logger.info(f"Stored job: {job_data.job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store job: {str(e)}")
            return False
    
    def store_scoring_result(self, scoring_result: ScoringResult) -> bool:
        """
        Store scoring result in Pinecone.
        
        Args:
            scoring_result: Scoring result to store
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            # Create combined text for embedding
            combined_text = f"Score: {scoring_result.score}, Explanation: {scoring_result.explanation}"
            embedding = self._generate_embedding(combined_text)
            
            # Prepare metadata
            metadata = {
                "type": "scoring",
                "scoring_id": scoring_result.scoring_id,
                "candidate_id": scoring_result.candidate_id,
                "job_id": scoring_result.job_id,
                "score": scoring_result.score,
                "confidence": scoring_result.confidence,
                "model_used": scoring_result.model_used,
                "created_at": scoring_result.created_at.isoformat(),
                "explanation": scoring_result.explanation[:1000],  # Truncate for metadata
                **scoring_result.metadata
            }
            
            # Store in Pinecone
            vector_id = f"scoring_{scoring_result.scoring_id}"
            self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            
            self.logger.info(f"Stored scoring result: {scoring_result.scoring_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store scoring result: {str(e)}")
            return False
    
    def search_candidates(self, query: str, job_id: Optional[str] = None, 
                         min_score: Optional[int] = None, max_results: int = 10,
                         skills_filter: Optional[List[str]] = None,
                         location_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for candidates using vector similarity.
        
        Args:
            query: Search query
            job_id: Filter by job ID
            min_score: Minimum score filter
            max_results: Maximum number of results
            skills_filter: Filter by required skills
            location_filter: Filter by location
            
        Returns:
            List[Dict[str, Any]]: List of matching candidates with scores
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build filter
            filter_dict = {"type": "candidate"}
            if job_id:
                filter_dict["job_id"] = job_id
            if min_score is not None:
                filter_dict["score"] = {"$gte": min_score}
            if skills_filter:
                # Note: This is a simplified filter. For complex skill matching,
                # you might want to implement more sophisticated logic
                pass
            if location_filter:
                filter_dict["location"] = location_filter
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=max_results,
                include_metadata=True
            )
            
            # Process results
            candidates = []
            for match in results.matches:
                metadata = match.metadata
                candidate = {
                    "candidate_id": metadata.get("candidate_id"),
                    "name": metadata.get("name"),
                    "email": metadata.get("email"),
                    "skills": json.loads(metadata.get("skills", "[]")),
                    "experience_years": metadata.get("experience_years"),
                    "current_role": metadata.get("current_role"),
                    "location": metadata.get("location"),
                    "status": metadata.get("status"),
                    "similarity_score": match.score,
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                }
                candidates.append(candidate)
            
            self.logger.info(f"Found {len(candidates)} candidates for query: {query}")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Failed to search candidates: {str(e)}")
            return []
    
    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific candidate by ID.
        
        Args:
            candidate_id: Candidate ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Candidate data if found
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            vector_id = f"candidate_{candidate_id}"
            results = self.index.fetch(ids=[vector_id])
            
            if vector_id in results.vectors:
                vector = results.vectors[vector_id]
                metadata = vector.metadata
                
                return {
                    "candidate_id": metadata.get("candidate_id"),
                    "name": metadata.get("name"),
                    "email": metadata.get("email"),
                    "skills": json.loads(metadata.get("skills", "[]")),
                    "experience_years": metadata.get("experience_years"),
                    "current_role": metadata.get("current_role"),
                    "location": metadata.get("location"),
                    "status": metadata.get("status"),
                    "profile_text": metadata.get("profile_text"),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get candidate {candidate_id}: {str(e)}")
            return None
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific job by ID.
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Job data if found
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            vector_id = f"job_{job_id}"
            results = self.index.fetch(ids=[vector_id])
            
            if vector_id in results.vectors:
                vector = results.vectors[vector_id]
                metadata = vector.metadata
                
                return {
                    "job_id": metadata.get("job_id"),
                    "title": metadata.get("title"),
                    "company": metadata.get("company"),
                    "required_skills": json.loads(metadata.get("required_skills", "[]")),
                    "preferred_skills": json.loads(metadata.get("preferred_skills", "[]")),
                    "experience_required": metadata.get("experience_required"),
                    "location": metadata.get("location"),
                    "salary_range": metadata.get("salary_range"),
                    "is_active": metadata.get("is_active"),
                    "description": metadata.get("description"),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {str(e)}")
            return None
    
    def get_scoring_history(self, candidate_id: Optional[str] = None, 
                           job_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve scoring history.
        
        Args:
            candidate_id: Filter by candidate ID
            job_id: Filter by job ID
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of scoring results
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            # Build filter
            filter_dict = {"type": "scoring"}
            if candidate_id:
                filter_dict["candidate_id"] = candidate_id
            if job_id:
                filter_dict["job_id"] = job_id
            
            # Query all scoring results (this is a simplified approach)
            # In production, you might want to use a different strategy
            results = self.index.query(
                vector=[0.0] * self.vector_dimension,  # Dummy vector
                filter=filter_dict,
                top_k=limit,
                include_metadata=True
            )
            
            scoring_results = []
            for match in results.matches:
                metadata = match.metadata
                result = {
                    "scoring_id": metadata.get("scoring_id"),
                    "candidate_id": metadata.get("candidate_id"),
                    "job_id": metadata.get("job_id"),
                    "score": metadata.get("score"),
                    "confidence": metadata.get("confidence"),
                    "model_used": metadata.get("model_used"),
                    "explanation": metadata.get("explanation"),
                    "created_at": metadata.get("created_at"),
                    "similarity_score": match.score
                }
                scoring_results.append(result)
            
            return scoring_results
            
        except Exception as e:
            self.logger.error(f"Failed to get scoring history: {str(e)}")
            return []
    
    def delete_candidate(self, candidate_id: str) -> bool:
        """
        Delete a candidate from Pinecone.
        
        Args:
            candidate_id: Candidate ID to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            vector_id = f"candidate_{candidate_id}"
            self.index.delete(ids=[vector_id])
            
            self.logger.info(f"Deleted candidate: {candidate_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete candidate {candidate_id}: {str(e)}")
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from Pinecone.
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            vector_id = f"job_{job_id}"
            self.index.delete(ids=[vector_id])
            
            self.logger.info(f"Deleted job: {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get Pinecone index statistics.
        
        Returns:
            Dict[str, Any]: Index statistics
        """
        try:
            if not self.is_connected:
                raise RuntimeError("Pinecone not connected")
            
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {str(e)}")
            return {}


# Global service instance
pinecone_service = PineconeService() 