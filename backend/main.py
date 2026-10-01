from fastapi import FastAPI

app = FastAPI(
    title="Gold Analysis API",
    description="XAUUSD Economic Events Analysis Backend",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Gold Analysis API is running"
    }


@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "engine": "Python",
        "market": "XAUUSD"
    }
