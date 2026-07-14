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


def update_llm_settings(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> None:
    """
    Hot-swap LLM credentials on the live Settings singleton and persist them
    to .env. get_settings() is lru_cache'd, so every module that already
    called it (llm_client.py, runner.py, qa_generator.py, routes.py,
    dedup.py, chunker.py) holds a reference to this same object — mutating
    its fields here takes effect immediately for any in-flight or future
    call, with no process restart needed.
    """
    from dotenv import set_key

    s = get_settings()
    if api_key:
        s.llm_api_key = api_key
        set_key(".env", "LLM_API_KEY", api_key)
    if base_url:
        s.llm_base_url = base_url
        set_key(".env", "LLM_BASE_URL", base_url)
    if model:
        s.llm_model = model
        set_key(".env", "LLM_MODEL", model)
