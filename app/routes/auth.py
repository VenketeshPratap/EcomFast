# app/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from google.oauth2 import id_token
from google.auth.transport import requests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import requests as httpx

from app.core.config import get_settings
from app.database import get_db
from app.models.user import User
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])
from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


@router.get("/google/login")
def google_login():
    settings = get_settings()  # ✅ CALL the function
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
        "&scope=openid%20email%20profile"
    )
    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    settings = get_settings()  # ✅ CALL it here too

    # Exchange code for token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    response = httpx.post(token_url, data=data)
    token_data = response.json()

    if "id_token" not in token_data:
        raise HTTPException(status_code=400, detail="Google token exchange failed")

    id_info = id_token.verify_oauth2_token(token_data["id_token"], requests.Request())

    email = id_info.get("email")
    name = id_info.get("name")

    # Check user exists or create
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(email=email, name=name)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    access_token = create_access_token({"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}
