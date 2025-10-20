from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # --- App Config ---
    APP_NAME: str = "EcomFast"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # --- Database ---
    DATABASE_URL: str | None = None
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # --- JWT / Auth ---
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # --- OAuth / Google ---
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str

    # --- Misc ---
    SECRET_KEY: str
    ALGORITHM: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
