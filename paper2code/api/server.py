"""
FastAPI application server for Paper2Code
"""

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.
    """
    app = FastAPI(
        title="Paper2Code API",
        description="Transform research papers into working code via HTTP API",
        version="1.0.0",
    )

    # Basic CORS (customize as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API router
    app.include_router(router)

    # Serve static frontend if built (frontend/dist)
    try:
        app.mount(
            "/",
            StaticFiles(directory="frontend/dist", html=True),
            name="static",
        )
    except Exception:
        # Ignore if frontend not built yet
        pass

    @app.get("/health", tags=["system"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start a uvicorn server hosting the FastAPI app.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()


