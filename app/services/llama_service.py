"""
LLaMA 2 service for job profile scoring.
"""

import os
import re
import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from loguru import logger

from app.utils.logger import get_logger
from app.models import ScoringResponse


class LLaMAService:
    """Service class for LLaMA 2 model operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_id = os.getenv("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf")
        self.is_loaded = False
        
        # Model generation parameters
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "200"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        
    def load_model(self) -> bool:
        """
        Load the LLaMA 2 model and tokenizer.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading LLaMA 2 model: {self.model_id}")
            
            # Check for Hugging Face token
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_auth_token=token,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory optimization
            self.logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=token,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True  # Enable 8-bit quantization for memory efficiency
            )
            
            # Create pipeline
            self.logger.info("Creating pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            self.is_loaded = True
            self.logger.info("LLaMA 2 model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LLaMA 2 model: {str(e)}")
            self.is_loaded = False
            return False
    
    def _create_scoring_prompt(self, candidate_profile: str, job_description: str) -> str:
        """
        Create a structured prompt for profile scoring.
        
        Args:
            candidate_profile: Candidate's profile text
            job_description: Job description text
            
        Returns:
            str: Formatted prompt for the model
        """
        prompt = f"""<s>[INST] You are an expert hiring assistant with deep knowledge of job matching and candidate evaluation.

Your task is to analyze how well a candidate's profile matches a specific job description and provide:
1. A numerical score from 0 to 100 (where 100 is a perfect match)
2. A detailed explanation of your reasoning
3. A confidence level in your assessment

Please be objective, thorough, and consider:
- Technical skills alignment
- Experience level match
- Domain expertise relevance
- Leadership/soft skills fit
- Overall suitability

Candidate Profile:
{candidate_profile}

Job Description:
{job_description}

Please provide your analysis in the following format:
Score: [0-100]
Explanation: [Detailed reasoning]
Confidence: [0.0-1.0] [/INST]"""
        
        return prompt
    
    def _extract_score_from_response(self, response: str) -> Tuple[int, str, float]:
        """
        Extract score, explanation, and confidence from model response.
        
        Args:
            response: Raw model response text
            
        Returns:
            Tuple[int, str, float]: (score, explanation, confidence)
        """
        try:
            # Extract score
            score_match = re.search(r'Score:\s*(\d+)', response, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 50
            
            # Extract explanation
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=Confidence:|$)', 
                                        response, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "Analysis completed"
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            # Validate ranges
            score = max(0, min(100, score))
            confidence = max(0.0, min(1.0, confidence))
            
            return score, explanation, confidence
            
        except Exception as e:
            self.logger.warning(f"Failed to parse model response: {str(e)}")
            return 50, "Unable to parse detailed response", 0.5
    
    def score_profile(self, candidate_profile: str, job_description: str) -> ScoringResponse:
        """
        Score a candidate profile against a job description.
        
        Args:
            candidate_profile: Candidate's profile text
            job_description: Job description text
            
        Returns:
            ScoringResponse: Scoring results with score, explanation, and metadata
            
        Raises:
            RuntimeError: If model is not loaded or scoring fails
        """
        if not self.is_loaded:
            raise RuntimeError("LLaMA 2 model is not loaded")
        
        try:
            self.logger.info("Starting profile scoring...")
            
            # Create prompt
            prompt = self._create_scoring_prompt(candidate_profile, job_description)
            
            # Generate response
            self.logger.info("Generating model response...")
            response = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the original prompt to get only the response
            model_response = generated_text[len(prompt):].strip()
            
            self.logger.info("Extracting scoring results...")
            score, explanation, confidence = self._extract_score_from_response(model_response)
            
            # Create response
            scoring_response = ScoringResponse(
                score=score,
                explanation=explanation,
                confidence=confidence,
                model_used=self.model_id
            )
            
            self.logger.info(f"Scoring completed. Score: {score}, Confidence: {confidence}")
            return scoring_response
            
        except Exception as e:
            self.logger.error(f"Profile scoring failed: {str(e)}")
            raise RuntimeError(f"Failed to score profile: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        return {
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "device": str(self.model.device) if self.model else None,
            "dtype": str(self.model.dtype) if self.model else None,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            self.is_loaded = False
            self.logger.info("Model unloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to unload model: {str(e)}")


# Global service instance
llama_service = LLaMAService() 