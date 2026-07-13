from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    n_questions_per_chunk: int = 5
    m_variations_per_question: int = 3
    dedup_similarity_threshold: float = 0.85
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 75
    db_path: str = "db/ift.db"
    data_output_dir: str = "data/output"

    # utf-8-sig transparently strips a leading BOM if present (e.g. a .env
    # file saved by certain Windows editors) and is a no-op otherwise.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig")


@lru_cache
def get_settings() -> Settings:
    return Settings()
