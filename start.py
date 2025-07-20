#!/usr/bin/env python3
"""
Startup script for ProfileScore application.
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

# Create logs directory if it doesn't exist
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"Starting ProfileScore API on {host}:{port}")
    print(f"Log level: {log_level}")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=True,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print("\nShutting down ProfileScore API...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1) 