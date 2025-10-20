# app/routes/users.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.deps import get_current_user
from app.database import get_db
from app.schemas.user import UserRead

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/me", response_model=UserRead)
async def read_users_me(current_user = Depends(get_current_user)):
    return current_user
