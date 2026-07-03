from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.database import create_db_and_tables
from app.api import routes, sse

app = FastAPI(title="IFT Dataset Builder", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


app.include_router(routes.router)
app.include_router(sse.router)

# Serve the minimal frontend at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
