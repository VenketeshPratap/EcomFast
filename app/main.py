# app/main.py
from fastapi import FastAPI, Depends
from app.database import engine, Base
from app.routes import auth, users
from app.core.config import get_settings
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

app.include_router(auth.router)
app.include_router(users.router)

@app.on_event("startup")
async def startup_event():
    # create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}

@app.get("/")
async def root():
    return {"message": "EcomFast API is running 🚀"}

bearer_scheme = HTTPBearer()

# Example protected route
@app.get("/protected")
def protected_route(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    # verify JWT token here
    return {"token": token}


