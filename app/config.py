from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    llm_api_key: str = "dummy"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    n_questions_per_chunk: int = 5
    m_variations_per_question: int = 3
    dedup_similarity_threshold: float = 0.85
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 75
    db_path: str = "db/ift.db"
    data_output_dir: str = "data/output"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
