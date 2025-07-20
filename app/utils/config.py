"""
Configuration validation for ProfileScore application.
"""

import os
from typing import Dict, List, Optional
from app.utils.logger import get_logger


class ConfigValidator:
    """Configuration validator for ProfileScore."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_required_env_vars(self) -> Dict[str, bool]:
        """
        Validate that all required environment variables are set.
        
        Returns:
            Dict[str, bool]: Dictionary with validation results
        """
        required_vars = {
            "HUGGINGFACE_TOKEN": "Required for LLaMA 2 model access",
            "PINECONE_API_KEY": "Required for vector storage",
            "PINECONE_ENVIRONMENT": "Required for Pinecone connection"
        }
        
        validation_results = {}
        
        for var_name, description in required_vars.items():
            value = os.getenv(var_name)
            is_valid = value is not None and value.strip() != ""
            validation_results[var_name] = is_valid
            
            if not is_valid:
                self.logger.warning(f"Missing required environment variable: {var_name} - {description}")
            else:
                self.logger.info(f"Environment variable validated: {var_name}")
        
        return validation_results
    
    def validate_optional_env_vars(self) -> Dict[str, bool]:
        """
        Validate optional environment variables with defaults.
        
        Returns:
            Dict[str, bool]: Dictionary with validation results
        """
        optional_vars = {
            "MODEL_ID": "meta-llama/Llama-2-7b-chat-hf",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "PINECONE_INDEX_NAME": "profilescores",
            "MAX_NEW_TOKENS": "200",
            "TEMPERATURE": "0.7",
            "TOP_P": "0.9",
            "LOG_LEVEL": "INFO",
            "HOST": "0.0.0.0",
            "PORT": "8000"
        }
        
        validation_results = {}
        
        for var_name, default_value in optional_vars.items():
            value = os.getenv(var_name, default_value)
            is_valid = value is not None and value.strip() != ""
            validation_results[var_name] = is_valid
            
            if is_valid:
                self.logger.info(f"Optional environment variable set: {var_name} = {value}")
            else:
                self.logger.warning(f"Optional environment variable missing: {var_name}, using default: {default_value}")
        
        return validation_results
    
    def get_config_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dict[str, any]: Configuration summary
        """
        return {
            "model": {
                "model_id": os.getenv("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "200")),
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("TOP_P", "0.9"))
            },
            "pinecone": {
                "index_name": os.getenv("PINECONE_INDEX_NAME", "profilescores"),
                "environment": os.getenv("PINECONE_ENVIRONMENT"),
                "api_key_set": bool(os.getenv("PINECONE_API_KEY"))
            },
            "application": {
                "host": os.getenv("HOST", "0.0.0.0"),
                "port": int(os.getenv("PORT", "8000")),
                "log_level": os.getenv("LOG_LEVEL", "INFO")
            },
            "huggingface": {
                "token_set": bool(os.getenv("HUGGINGFACE_TOKEN"))
            }
        }


# Global validator instance
config_validator = ConfigValidator() 