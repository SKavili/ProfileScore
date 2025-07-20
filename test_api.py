"""
Test script for ProfileScore API.
"""

import requests
import json
import time
from typing import Dict, Any


class ProfileScoreTester:
    """Test client for ProfileScore API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> Dict[str, Any]:
        """Test the health endpoint."""
        print("Testing health endpoint...")
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test the model info endpoint."""
        print("Testing model info endpoint...")
        response = self.session.get(f"{self.base_url}/model/info")
        return response.json()
    
    def test_scoring(self, candidate_profile: str, job_description: str) -> Dict[str, Any]:
        """Test the profile scoring endpoint."""
        print("Testing profile scoring endpoint...")
        
        payload = {
            "candidate_profile": candidate_profile,
            "job_description": job_description
        }
        
        response = self.session.post(
            f"{self.base_url}/score",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def run_full_test(self):
        """Run a complete test suite."""
        print("=" * 60)
        print("ProfileScore API Test Suite")
        print("=" * 60)
        
        # Test 1: Health check
        try:
            health_result = self.test_health()
            print(f"✅ Health Check: {health_result}")
        except Exception as e:
            print(f"❌ Health Check Failed: {e}")
            return
        
        # Test 2: Model info
        try:
            model_info = self.test_model_info()
            print(f"✅ Model Info: {json.dumps(model_info, indent=2)}")
        except Exception as e:
            print(f"❌ Model Info Failed: {e}")
        
        # Test 3: Profile scoring
        candidate_profile = """
        Senior Software Engineer with 7 years of experience in backend development using Python, Django, and AWS.
        Worked on scalable microservices and RESTful APIs in fintech and health-tech domains.
        Led a team of 5 developers, introduced CI/CD pipelines, and optimized DB queries.
        Strong experience with PostgreSQL, Redis, and Docker. Familiar with React and Node.js.
        """
        
        job_description = """
        We are looking for a Backend Engineer with 5+ years of experience in Python, Django/Flask.
        Must have experience deploying on AWS, building microservices, and working with REST APIs.
        Leadership experience is a plus. Knowledge of databases and DevOps practices required.
        Experience with frontend technologies is beneficial but not required.
        """
        
        try:
            scoring_result = self.test_scoring(candidate_profile, job_description)
            print(f"✅ Profile Scoring: {json.dumps(scoring_result, indent=2)}")
        except Exception as e:
            print(f"❌ Profile Scoring Failed: {e}")
        
        print("=" * 60)
        print("Test suite completed!")


def main():
    """Main function to run the test suite."""
    tester = ProfileScoreTester()
    
    # Wait a bit for the server to start if needed
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    tester.run_full_test()


if __name__ == "__main__":
    main() 