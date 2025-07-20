#!/usr/bin/env python3
"""
Architecture validation test for ProfileScore.
This script tests all components to ensure the architecture is sound.
"""

import os
import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        "app",
        "app.main",
        "app.models",
        "app.services.llama_service",
        "app.services.pinecone_service", 
        "app.services.langchain_service",
        "app.utils.logger",
        "app.utils.config"
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} modules")
        return False
    else:
        print(f"\n✅ All {len(modules_to_test)} modules imported successfully")
        return True

def test_models():
    """Test that all Pydantic models are properly defined."""
    print("\n🔍 Testing Pydantic models...")
    
    try:
        from app.models import (
            ScoringRequest, ScoringResponse, HealthResponse, ErrorResponse,
            CandidateData, JobData, ScoringResult, CandidateSearchRequest,
            CandidateSearchResponse, CandidateStatus
        )
        
        # Test model instantiation
        test_request = ScoringRequest(
            candidate_profile="Test candidate profile",
            job_description="Test job description"
        )
        
        test_response = ScoringResponse(
            score=85,
            explanation="Test explanation",
            confidence=0.8,
            model_used="test-model"
        )
        
        print("  ✅ All models imported and instantiated successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def test_services():
    """Test that all services can be instantiated."""
    print("\n🔍 Testing service instantiation...")
    
    try:
        from app.services.llama_service import llama_service
        from app.services.pinecone_service import pinecone_service
        from app.services.langchain_service import langchain_service
        
        # Test service attributes
        assert hasattr(llama_service, 'load_model')
        assert hasattr(pinecone_service, 'connect')
        assert hasattr(langchain_service, 'initialize')
        
        print("  ✅ All services instantiated successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Service test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\n🔍 Testing utility functions...")
    
    try:
        from app.utils.logger import get_logger, setup_logger
        from app.utils.config import config_validator
        
        # Test logger
        logger = get_logger("test")
        assert logger is not None
        
        # Test config validator
        assert hasattr(config_validator, 'validate_required_env_vars')
        assert hasattr(config_validator, 'validate_optional_env_vars')
        
        print("  ✅ All utilities working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Utility test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "transformers",
        "torch",
        "pinecone",
        "langchain",
        "sentence-transformers",
        "loguru",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "python-dotenv":
                importlib.import_module("dotenv")
            else:
                importlib.import_module(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print(f"\n✅ All {len(required_packages)} dependencies available")
        return True

def test_architecture_patterns():
    """Test architectural patterns and design principles."""
    print("\n🔍 Testing architectural patterns...")
    
    patterns_valid = True
    
    # Test 1: Service pattern
    try:
        from app.services.llama_service import LLaMAService
        from app.services.pinecone_service import PineconeService
        from app.services.langchain_service import LangChainService
        
        # Check if services follow singleton pattern
        llama1 = LLaMAService()
        llama2 = LLaMAService()
        assert llama1 is not llama2  # Should be different instances
        
        print("  ✅ Service pattern: Multiple instances allowed")
    except Exception as e:
        print(f"  ❌ Service pattern test failed: {e}")
        patterns_valid = False
    
    # Test 2: Model validation
    try:
        from app.models import ScoringRequest
        
        # Test validation
        try:
            ScoringRequest(candidate_profile="", job_description="test")
            print("  ❌ Model validation: Should reject empty profile")
            patterns_valid = False
        except:
            print("  ✅ Model validation: Properly validates input")
            
    except Exception as e:
        print(f"  ❌ Model validation test failed: {e}")
        patterns_valid = False
    
    # Test 3: Error handling
    try:
        from app.models import ErrorResponse
        
        error_response = ErrorResponse(
            error="Test error",
            detail="Test detail"
        )
        assert error_response.error == "Test error"
        print("  ✅ Error handling: ErrorResponse model works")
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        patterns_valid = False
    
    return patterns_valid

def test_file_structure():
    """Test that the file structure is correct."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "app/__init__.py",
        "app/main.py", 
        "app/models.py",
        "app/services/__init__.py",
        "app/services/llama_service.py",
        "app/services/pinecone_service.py",
        "app/services/langchain_service.py",
        "app/utils/__init__.py",
        "app/utils/logger.py",
        "app/utils/config.py",
        "requirements.txt",
        "start.py",
        "README.md",
        "env.example"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files present")
        return True

def main():
    """Run all architecture tests."""
    print("🏗️  ProfileScore Architecture Validation Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Pydantic Models", test_models),
        ("Services", test_services),
        ("Utilities", test_utils),
        ("Architectural Patterns", test_architecture_patterns)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Architecture is sound.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 