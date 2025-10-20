# app/crud/user.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.models.user import User
from app.schemas.user import UserCreate



    


class CRUDUser:
    async def get_by_id(self, db: AsyncSession, user_id: int) -> Optional[User]:
        q = select(User).where(User.id == user_id)
        res = await db.execute(q)
        return res.scalars().first()

    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        q = select(User).where(User.email == email)
        res = await db.execute(q)
        return res.scalars().first()
    # app/crud/user.py

    async def create_google_user(self, db: AsyncSession, email: str, name: str, google_sub: str) -> User:
        user = User(email=email, name=name, google_sub=google_sub)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    # async def create(self, db: AsyncSession, *, obj_in: UserCreate) -> User:
    #     hashed_password = get_password_hash(obj_in.password)
    #     user = User(email=obj_in.email, name=obj_in.name, hashed_password=hashed_password)
    #     db.add(user)
    #     await db.commit()
    #     await db.refresh(user)
    #     return user

    async def authenticate(self, db: AsyncSession, email: str, plain_password: str, verify_fn) -> Optional[User]:
        user = await self.get_by_email(db, email=email)
        if not user or not user.hashed_password:
            return None
        if not verify_fn(plain_password, user.hashed_password):
            return None
        return user

user = CRUDUser()


