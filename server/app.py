# server/app.py — content-moderation-openenv

import uvicorn
from main import app


def main():
    """
    Entry point for the Content Moderation OpenEnv server.
    Called by: python -m server.app or via 'serve' console script
    Runs the FastAPI application on 0.0.0.0:7860.
    """
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


# Alias for compatibility with multiple entry points
start_server = main


if __name__ == "__main__":
    main()
