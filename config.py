import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for Content Moderation OpenEnv."""
    
    # API Configuration
    API_BASE_URL: str = os.getenv(
        "API_BASE_URL",
        "http://localhost:7860"
    )
    
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME",
        "meta-llama/Llama-2-7b-hf"
    )
    
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    
    # Server Configuration
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    
    # Environment Configuration
    MAX_STEPS_PER_EPISODE: int = 10
    NUM_TASKS: int = 3
    
    @classmethod
    def validate(cls):
        """Validate configuration and warn about missing values."""
        logger.info("Validating configuration...")
        
        warnings = []
        
        if not cls.API_BASE_URL:
            warnings.append("API_BASE_URL not set, using default: http://localhost:7860")
        
        if not cls.MODEL_NAME:
            warnings.append("MODEL_NAME not set, using default: meta-llama/Llama-2-7b-hf")
        
        if not cls.HF_TOKEN:
            warnings.append("HF_TOKEN not set. This may be required for Hugging Face API access.")
        
        for warning in warnings:
            logger.warning(f" {warning}")
        
        if warnings:
            logger.info(f"Total warnings: {len(warnings)}")
        else:
            logger.info("[OK] All configuration values are set correctly")
        
        logger.info(f"API_BASE_URL: {cls.API_BASE_URL}")
        logger.info(f"MODEL_NAME: {cls.MODEL_NAME}")
        logger.info(f"Server: {cls.SERVER_HOST}:{cls.SERVER_PORT}")
    
    @classmethod
    def to_dict(cls):
        """Return configuration as dictionary."""
        return {
            "API_BASE_URL": cls.API_BASE_URL,
            "MODEL_NAME": cls.MODEL_NAME,
            "HF_TOKEN": "***" if cls.HF_TOKEN else None,
            "SERVER_HOST": cls.SERVER_HOST,
            "SERVER_PORT": cls.SERVER_PORT,
            "MAX_STEPS_PER_EPISODE": cls.MAX_STEPS_PER_EPISODE,
            "NUM_TASKS": cls.NUM_TASKS,
        }


# Initialize and validate config on import
try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    raise
