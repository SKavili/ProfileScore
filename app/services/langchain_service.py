"""
LangChain service for enhanced AI processing and analysis.
"""

import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.schema import Document
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from app.utils.logger import get_logger
from app.models import CandidateData, JobData, ScoringResult
from app.services.llama_service import llama_service


class LangChainService:
    """Service class for LangChain-based AI processing."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.llm = None
        self.embeddings = None
        self.text_splitter = None
        self.vectorstore = None
        self.is_initialized = False
        
        # Configuration
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def initialize(self) -> bool:
        """
        Initialize LangChain components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing LangChain service...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize LLM from LLaMA service
            if llama_service.is_loaded:
                self.llm = HuggingFacePipeline(
                    pipeline=llama_service.pipeline,
                    model_kwargs={"temperature": 0.7, "max_length": 200}
                )
            
            self.is_initialized = True
            self.logger.info("LangChain service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain service: {str(e)}")
            self.is_initialized = False
            return False
    
    def extract_skills(self, profile_text: str) -> List[str]:
        """
        Extract skills from candidate profile using LangChain.
        
        Args:
            profile_text: Candidate profile text
            
        Returns:
            List[str]: Extracted skills
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            prompt_template = PromptTemplate(
                input_variables=["profile_text"],
                template="""
                Extract technical skills, programming languages, frameworks, tools, and technologies from the following candidate profile.
                Return only a JSON array of skill names, without any additional text.
                
                Profile: {profile_text}
                
                Skills:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(profile_text=profile_text)
            
            # Parse JSON result
            try:
                skills = json.loads(result.strip())
                if isinstance(skills, list):
                    return [skill.strip() for skill in skills if skill.strip()]
            except json.JSONDecodeError:
                # Fallback: extract skills using simple text processing
                return self._fallback_skill_extraction(profile_text)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to extract skills: {str(e)}")
            return self._fallback_skill_extraction(profile_text)
    
    def _fallback_skill_extraction(self, profile_text: str) -> List[str]:
        """
        Fallback skill extraction using simple text processing.
        
        Args:
            profile_text: Candidate profile text
            
        Returns:
            List[str]: Extracted skills
        """
        # Common technical skills and technologies
        common_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI",
            "Spring", "Express.js", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Kafka",
            "Git", "Jenkins", "CI/CD", "REST API", "GraphQL", "Microservices",
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
            "Data Analysis", "SQL", "NoSQL", "Big Data", "Hadoop", "Spark"
        ]
        
        profile_lower = profile_text.lower()
        extracted_skills = []
        
        for skill in common_skills:
            if skill.lower() in profile_lower:
                extracted_skills.append(skill)
        
        return extracted_skills
    
    def analyze_profile(self, profile_text: str) -> Dict[str, Any]:
        """
        Analyze candidate profile using LangChain.
        
        Args:
            profile_text: Candidate profile text
            
        Returns:
            Dict[str, Any]: Profile analysis results
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            prompt_template = PromptTemplate(
                input_variables=["profile_text"],
                template="""
                Analyze the following candidate profile and extract key information.
                Return the result as a JSON object with the following structure:
                {{
                    "experience_years": <estimated years of experience>,
                    "current_role": "<current job title>",
                    "location": "<location if mentioned>",
                    "education": "<education level if mentioned>",
                    "key_achievements": ["<achievement1>", "<achievement2>"],
                    "industries": ["<industry1>", "<industry2>"],
                    "summary": "<brief professional summary>"
                }}
                
                Profile: {profile_text}
                
                Analysis:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(profile_text=profile_text)
            
            # Parse JSON result
            try:
                analysis = json.loads(result.strip())
                return analysis
            except json.JSONDecodeError:
                return self._fallback_profile_analysis(profile_text)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze profile: {str(e)}")
            return self._fallback_profile_analysis(profile_text)
    
    def _fallback_profile_analysis(self, profile_text: str) -> Dict[str, Any]:
        """
        Fallback profile analysis using simple text processing.
        
        Args:
            profile_text: Candidate profile text
            
        Returns:
            Dict[str, Any]: Basic profile analysis
        """
        return {
            "experience_years": None,
            "current_role": None,
            "location": None,
            "education": None,
            "key_achievements": [],
            "industries": [],
            "summary": profile_text[:200] + "..." if len(profile_text) > 200 else profile_text
        }
    
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract job requirements from job description using LangChain.
        
        Args:
            job_description: Job description text
            
        Returns:
            Dict[str, Any]: Extracted job requirements
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            prompt_template = PromptTemplate(
                input_variables=["job_description"],
                template="""
                Extract job requirements from the following job description.
                Return the result as a JSON object with the following structure:
                {{
                    "required_skills": ["<skill1>", "<skill2>"],
                    "preferred_skills": ["<skill1>", "<skill2>"],
                    "experience_required": <years of experience required>,
                    "education_required": "<education level>",
                    "responsibilities": ["<responsibility1>", "<responsibility2>"],
                    "benefits": ["<benefit1>", "<benefit2>"],
                    "job_type": "<full-time/part-time/contract>",
                    "location": "<job location>",
                    "salary_range": "<salary range if mentioned>"
                }}
                
                Job Description: {job_description}
                
                Requirements:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(job_description=job_description)
            
            # Parse JSON result
            try:
                requirements = json.loads(result.strip())
                return requirements
            except json.JSONDecodeError:
                return self._fallback_job_requirements(job_description)
            
        except Exception as e:
            self.logger.error(f"Failed to extract job requirements: {str(e)}")
            return self._fallback_job_requirements(job_description)
    
    def _fallback_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Fallback job requirements extraction using simple text processing.
        
        Args:
            job_description: Job description text
            
        Returns:
            Dict[str, Any]: Basic job requirements
        """
        return {
            "required_skills": self.extract_skills(job_description),
            "preferred_skills": [],
            "experience_required": None,
            "education_required": None,
            "responsibilities": [],
            "benefits": [],
            "job_type": "full-time",
            "location": None,
            "salary_range": None
        }
    
    def enhanced_scoring(self, candidate_profile: str, job_description: str) -> Dict[str, Any]:
        """
        Enhanced scoring using LangChain with multiple analysis components.
        
        Args:
            candidate_profile: Candidate profile text
            job_description: Job description text
            
        Returns:
            Dict[str, Any]: Enhanced scoring results
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            # Extract candidate skills and analysis
            candidate_skills = self.extract_skills(candidate_profile)
            candidate_analysis = self.analyze_profile(candidate_profile)
            
            # Extract job requirements
            job_requirements = self.extract_job_requirements(job_description)
            
            # Calculate skill match
            required_skills = job_requirements.get("required_skills", [])
            preferred_skills = job_requirements.get("preferred_skills", [])
            
            required_match = len(set(candidate_skills) & set(required_skills))
            preferred_match = len(set(candidate_skills) & set(preferred_skills))
            
            skill_match_score = 0
            if required_skills:
                skill_match_score = (required_match / len(required_skills)) * 70
            if preferred_skills:
                skill_match_score += (preferred_match / len(preferred_skills)) * 30
            
            # Experience match
            experience_match = 0
            candidate_exp = candidate_analysis.get("experience_years")
            required_exp = job_requirements.get("experience_required")
            
            if candidate_exp and required_exp:
                if candidate_exp >= required_exp:
                    experience_match = 100
                else:
                    experience_match = max(0, (candidate_exp / required_exp) * 100)
            
            # Get base scoring from LLaMA
            base_scoring = llama_service.score_profile(candidate_profile, job_description)
            
            # Enhanced scoring with multiple factors
            enhanced_score = (
                base_scoring.score * 0.6 +  # Base AI score
                skill_match_score * 0.3 +   # Skill match
                experience_match * 0.1      # Experience match
            )
            
            return {
                "enhanced_score": int(enhanced_score),
                "base_score": base_scoring.score,
                "skill_match_score": int(skill_match_score),
                "experience_match_score": int(experience_match),
                "explanation": base_scoring.explanation,
                "confidence": base_scoring.confidence,
                "candidate_skills": candidate_skills,
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "skill_matches": {
                    "required": required_match,
                    "preferred": preferred_match,
                    "total_required": len(required_skills),
                    "total_preferred": len(preferred_skills)
                },
                "candidate_analysis": candidate_analysis,
                "job_requirements": job_requirements,
                "model_used": base_scoring.model_used
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced scoring failed: {str(e)}")
            # Fallback to basic scoring
            basic_result = llama_service.score_profile(candidate_profile, job_description)
            return {
                "enhanced_score": basic_result.score,
                "base_score": basic_result.score,
                "skill_match_score": 0,
                "experience_match_score": 0,
                "explanation": basic_result.explanation,
                "confidence": basic_result.confidence,
                "candidate_skills": [],
                "required_skills": [],
                "preferred_skills": [],
                "skill_matches": {"required": 0, "preferred": 0, "total_required": 0, "total_preferred": 0},
                "candidate_analysis": {},
                "job_requirements": {},
                "model_used": basic_result.model_used
            }
    
    def create_search_tools(self) -> List[Tool]:
        """
        Create LangChain tools for enhanced search capabilities.
        
        Returns:
            List[Tool]: List of search tools
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            # Skill search tool
            skill_search_tool = Tool(
                name="skill_search",
                description="Search for candidates with specific skills",
                func=self._search_by_skills
            )
            
            # Experience search tool
            experience_search_tool = Tool(
                name="experience_search",
                description="Search for candidates with minimum years of experience",
                func=self._search_by_experience
            )
            
            # Location search tool
            location_search_tool = Tool(
                name="location_search",
                description="Search for candidates in specific locations",
                func=self._search_by_location
            )
            
            return [skill_search_tool, experience_search_tool, location_search_tool]
            
        except Exception as e:
            self.logger.error(f"Failed to create search tools: {str(e)}")
            return []
    
    def _search_by_skills(self, skills_query: str) -> str:
        """Search candidates by skills."""
        # This would integrate with Pinecone service
        return f"Searching for candidates with skills: {skills_query}"
    
    def _search_by_experience(self, experience_query: str) -> str:
        """Search candidates by experience."""
        return f"Searching for candidates with experience: {experience_query}"
    
    def _search_by_location(self, location_query: str) -> str:
        """Search candidates by location."""
        return f"Searching for candidates in location: {location_query}"
    
    def create_agent(self) -> Any:
        """
        Create a LangChain agent for complex queries.
        
        Returns:
            Any: LangChain agent
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            tools = self.create_search_tools()
            memory = ConversationBufferMemory(memory_key="chat_history")
            
            agent = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True
            )
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {str(e)}")
            return None
    
    def process_batch(self, candidates: List[str], job_description: str) -> List[Dict[str, Any]]:
        """
        Process multiple candidates in batch for efficiency.
        
        Args:
            candidates: List of candidate profiles
            job_description: Job description to match against
            
        Returns:
            List[Dict[str, Any]]: Batch processing results
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LangChain service not initialized")
            
            results = []
            for i, candidate_profile in enumerate(candidates):
                self.logger.info(f"Processing candidate {i+1}/{len(candidates)}")
                
                try:
                    result = self.enhanced_scoring(candidate_profile, job_description)
                    result["candidate_index"] = i
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process candidate {i+1}: {str(e)}")
                    results.append({
                        "candidate_index": i,
                        "error": str(e),
                        "enhanced_score": 0
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return []


# Global service instance
langchain_service = LangChainService() 