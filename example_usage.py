#!/usr/bin/env python3
"""
Example usage of ProfileScore LLaMA service.
This script demonstrates how to use the service directly without the API.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.services.llama_service import LLaMAService
from app.utils.logger import get_logger


def main():
    """Main function demonstrating LLaMA service usage."""
    logger = get_logger(__name__)
    
    # Check for Hugging Face token
    if not os.getenv("HUGGINGFACE_TOKEN"):
        print("❌ Error: HUGGINGFACE_TOKEN environment variable is required")
        print("Please set it in your .env file or environment")
        return
    
    # Create service instance
    service = LLaMAService()
    
    # Load the model
    print("🔄 Loading LLaMA 2 model...")
    if not service.load_model():
        print("❌ Failed to load LLaMA 2 model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Example candidate profile and job description
    candidate_profile = """
    Senior Software Engineer with 7 years of experience in backend development using Python, Django, and AWS.
    Worked on scalable microservices and RESTful APIs in fintech and health-tech domains.
    Led a team of 5 developers, introduced CI/CD pipelines, and optimized DB queries.
    Strong experience with PostgreSQL, Redis, and Docker. Familiar with React and Node.js.
    Experience with machine learning projects and data analysis.
    """
    
    job_description = """
    We are looking for a Backend Engineer with 5+ years of experience in Python, Django/Flask.
    Must have experience deploying on AWS, building microservices, and working with REST APIs.
    Leadership experience is a plus. Knowledge of databases and DevOps practices required.
    Experience with frontend technologies is beneficial but not required.
    Bonus points for experience with machine learning or data science projects.
    """
    
    # Score the profile
    print("\n🔄 Scoring candidate profile against job description...")
    try:
        result = service.score_profile(candidate_profile, job_description)
        
        print("\n" + "="*60)
        print("PROFILE SCORING RESULTS")
        print("="*60)
        print(f"Score: {result.score}/100")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Model Used: {result.model_used}")
        print(f"Timestamp: {result.timestamp}")
        print(f"\nExplanation:\n{result.explanation}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during scoring: {e}")
    
    # Get model information
    print("\n📊 Model Information:")
    model_info = service.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Clean up
    print("\n🧹 Cleaning up...")
    service.unload_model()
    print("✅ Done!")


if __name__ == "__main__":
    main() 