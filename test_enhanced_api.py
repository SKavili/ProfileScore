"""
Enhanced test script for ProfileScore API with Pinecone and LangChain features.
"""

import requests
import json
import time
import uuid
from typing import Dict, Any


class EnhancedProfileScoreTester:
    """Enhanced test client for ProfileScore API."""
    
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
    
    def test_basic_scoring(self, candidate_profile: str, job_description: str) -> Dict[str, Any]:
        """Test the basic profile scoring endpoint."""
        print("Testing basic profile scoring endpoint...")
        
        payload = {
            "candidate_profile": candidate_profile,
            "job_description": job_description,
            "candidate_id": f"test_candidate_{uuid.uuid4().hex[:8]}",
            "job_id": f"test_job_{uuid.uuid4().hex[:8]}",
            "metadata": {
                "test_type": "basic_scoring",
                "timestamp": time.time()
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/score",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def test_enhanced_scoring(self, candidate_profile: str, job_description: str) -> Dict[str, Any]:
        """Test the enhanced scoring endpoint with LangChain."""
        print("Testing enhanced scoring endpoint...")
        
        payload = {
            "candidate_profile": candidate_profile,
            "job_description": job_description,
            "candidate_id": f"test_candidate_{uuid.uuid4().hex[:8]}",
            "job_id": f"test_job_{uuid.uuid4().hex[:8]}",
            "metadata": {
                "test_type": "enhanced_scoring",
                "timestamp": time.time()
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/score/enhanced",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def test_create_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test creating a candidate in Pinecone."""
        print("Testing candidate creation...")
        
        response = self.session.post(
            f"{self.base_url}/candidates",
            json=candidate_data,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def test_create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test creating a job in Pinecone."""
        print("Testing job creation...")
        
        response = self.session.post(
            f"{self.base_url}/jobs",
            json=job_data,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def test_search_candidates(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Test candidate search functionality."""
        print("Testing candidate search...")
        
        payload = {
            "query": query,
            "max_results": max_results,
            "min_score": 70
        }
        
        response = self.session.post(
            f"{self.base_url}/candidates/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def test_batch_scoring(self, candidates: list, job_description: str) -> Dict[str, Any]:
        """Test batch scoring functionality."""
        print("Testing batch scoring...")
        
        payload = {
            "candidates": candidates,
            "job_description": job_description
        }
        
        response = self.session.post(
            f"{self.base_url}/batch/score",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def test_scoring_history(self) -> Dict[str, Any]:
        """Test scoring history retrieval."""
        print("Testing scoring history...")
        
        response = self.session.get(f"{self.base_url}/scoring/history?limit=10")
        return response.json()
    
    def run_full_test(self):
        """Run a complete test suite for all features."""
        print("=" * 80)
        print("Enhanced ProfileScore API Test Suite")
        print("=" * 80)
        
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
        
        # Sample data
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
        
        # Test 3: Basic scoring
        try:
            basic_result = self.test_basic_scoring(candidate_profile, job_description)
            print(f"✅ Basic Scoring: Score {basic_result.get('score', 'N/A')}")
        except Exception as e:
            print(f"❌ Basic Scoring Failed: {e}")
        
        # Test 4: Enhanced scoring
        try:
            enhanced_result = self.test_enhanced_scoring(candidate_profile, job_description)
            print(f"✅ Enhanced Scoring: Score {enhanced_result.get('enhanced_score', 'N/A')}")
            print(f"   Skill Match: {enhanced_result.get('skill_match_score', 'N/A')}")
            print(f"   Experience Match: {enhanced_result.get('experience_match_score', 'N/A')}")
        except Exception as e:
            print(f"❌ Enhanced Scoring Failed: {e}")
        
        # Test 5: Create candidate
        try:
            candidate_data = {
                "candidate_id": f"test_candidate_{uuid.uuid4().hex[:8]}",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "profile_text": candidate_profile,
                "skills": ["Python", "Django", "AWS", "PostgreSQL"],
                "experience_years": 7,
                "current_role": "Senior Software Engineer",
                "location": "San Francisco, CA",
                "status": "active",
                "metadata": {
                    "source": "test",
                    "created_by": "test_script"
                }
            }
            
            candidate_result = self.test_create_candidate(candidate_data)
            print(f"✅ Candidate Creation: {candidate_result.get('message', 'N/A')}")
            candidate_id = candidate_data["candidate_id"]
        except Exception as e:
            print(f"❌ Candidate Creation Failed: {e}")
            candidate_id = None
        
        # Test 6: Create job
        try:
            job_data = {
                "job_id": f"test_job_{uuid.uuid4().hex[:8]}",
                "title": "Senior Backend Engineer",
                "company": "TechCorp",
                "description": job_description,
                "required_skills": ["Python", "Django", "AWS"],
                "preferred_skills": ["Machine Learning", "React"],
                "experience_required": 5,
                "location": "San Francisco, CA",
                "salary_range": "$120k - $180k",
                "is_active": True,
                "metadata": {
                    "source": "test",
                    "created_by": "test_script"
                }
            }
            
            job_result = self.test_create_job(job_data)
            print(f"✅ Job Creation: {job_result.get('message', 'N/A')}")
            job_id = job_data["job_id"]
        except Exception as e:
            print(f"❌ Job Creation Failed: {e}")
            job_id = None
        
        # Test 7: Search candidates
        try:
            search_result = self.test_search_candidates("Python backend engineer", 5)
            print(f"✅ Candidate Search: Found {search_result.get('total_results', 0)} candidates")
        except Exception as e:
            print(f"❌ Candidate Search Failed: {e}")
        
        # Test 8: Batch scoring
        try:
            batch_candidates = [
                "Junior developer with 2 years of Python experience",
                "Senior engineer with 8 years of full-stack development",
                "Data scientist with Python and machine learning background"
            ]
            
            batch_result = self.test_batch_scoring(batch_candidates, job_description)
            print(f"✅ Batch Scoring: Processed {batch_result.get('total_processed', 0)} candidates")
            print(f"   Successful: {batch_result.get('successful', 0)}, Failed: {batch_result.get('failed', 0)}")
        except Exception as e:
            print(f"❌ Batch Scoring Failed: {e}")
        
        # Test 9: Scoring history
        try:
            history_result = self.test_scoring_history()
            print(f"✅ Scoring History: Retrieved {history_result.get('total_results', 0)} results")
        except Exception as e:
            print(f"❌ Scoring History Failed: {e}")
        
        # Test 10: Get specific candidate/job (if created successfully)
        if candidate_id:
            try:
                response = self.session.get(f"{self.base_url}/candidates/{candidate_id}")
                if response.status_code == 200:
                    print(f"✅ Candidate Retrieval: Successfully retrieved candidate {candidate_id}")
                else:
                    print(f"⚠️ Candidate Retrieval: Status {response.status_code}")
            except Exception as e:
                print(f"❌ Candidate Retrieval Failed: {e}")
        
        if job_id:
            try:
                response = self.session.get(f"{self.base_url}/jobs/{job_id}")
                if response.status_code == 200:
                    print(f"✅ Job Retrieval: Successfully retrieved job {job_id}")
                else:
                    print(f"⚠️ Job Retrieval: Status {response.status_code}")
            except Exception as e:
                print(f"❌ Job Retrieval Failed: {e}")
        
        print("=" * 80)
        print("Enhanced test suite completed!")
        print("=" * 80)


def main():
    """Main function to run the enhanced test suite."""
    tester = EnhancedProfileScoreTester()
    
    # Wait a bit for the server to start if needed
    print("Waiting for server to be ready...")
    time.sleep(3)
    
    tester.run_full_test()


if __name__ == "__main__":
    main() 