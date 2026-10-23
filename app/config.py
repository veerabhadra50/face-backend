"""Application configuration loaded from environment variables."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://postgres:password@localhost:5432/faceattendance"
    # Supabase project credentials (optional — only needed if you use the
    # Supabase REST/JS client or expose the project to the frontend).
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    face_match_threshold: float = 0.45
    face_det_threshold: float = 0.5
    insightface_model: str = "buffalo_l"
    cors_origins: str = "*"
    # Absolute path to the frontend folder to serve as static files.
    # Leave empty to run the API without serving the UI.
    frontend_dir: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
