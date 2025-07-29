#!/usr/bin/env python3
"""
Test script to verify LLaMA service fixes.
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


def test_llama_service():
    """Test the LLaMA service with the fixes."""
    logger = get_logger(__name__)
    
    print("=" * 60)
    print("Testing LLaMA Service Fixes")
    print("=" * 60)
    
    # Check environment variables
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token or token == "your_huggingface_token_here":
        print("❌ HUGGINGFACE_TOKEN not set or using placeholder value")
        print("Please set your actual Hugging Face token in the .env file")
        return False
    
    print(f"✅ HUGGINGFACE_TOKEN is set")
    
    # Test model info method (this was causing the frozenset error)
    print("\nTesting model info method...")
    service = LLaMAService()
    
    try:
        model_info = service.get_model_info()
        print(f"✅ Model info retrieved successfully: {model_info}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False
    
    # Test model loading (this was causing the quantization issues)
    print("\nTesting model loading...")
    try:
        success = service.load_model()
        if success:
            print("✅ Model loaded successfully")
            
            # Test model info after loading
            model_info = service.get_model_info()
            print(f"✅ Model info after loading: {model_info}")
            
            # Test unload
            service.unload_model()
            print("✅ Model unloaded successfully")
            
            return True
        else:
            print("❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed with exception: {e}")
        return False


def main():
    """Main function."""
    success = test_llama_service()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All LLaMA service fixes are working correctly!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some issues remain. Please check the error messages above.")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    main() 