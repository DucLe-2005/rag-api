from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # MongoDB configs
    MONGO_DATABASE_URI: str = "mongodb+srv://ducleanh2005:123Tienlen!123@ducle.wfkzpoe.mongodb.net/?retryWrites=true&w=majority&appName=ducle"
    MONGO_DATABASE_NAME: str = "ducle"

    # MQ config
    RABBITMQ_DEFAULT_USERNAME: str = "guest"
    RABBITMQ_DEFAULT_PASSWORD: str = "guest"
    RABBITMQ_HOST: str = "mq"
    RABBITMQ_PORT: int = 5673

    # QdrantDB config
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    USE_QDRANT_CLOUD: bool = True
    QDRANT_CLOUD_URL: str = "https://5b7a779e-5532-4bc2-9cad-649ad8f8d034.us-east-1-0.aws.cloud.qdrant.io"
    QDRANT_APIKEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.pxp_du1nrHJJhAcPgFXhoQjZfsx8TLW5Blw3WhE3Qn0"
   

    # OpenAI config
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # Groq config
    GROQ_API_KEY: str | None = None
    GROQ_MODEL_ID: str = "llama3-8b-8192"

    # CometML config
    COMET_API_KEY: str | None = None
    COMET_WORKSPACE: str | None = None
    COMET_PROJECT: str = "llm-twin"

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # LLM Model config
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_ID: str = "pauliusztin/LLMTwin-Llama-3.1-8B"
    DEPLOYMENT_ENDPOINT_NAME: str = "finbud"

    MAX_INPUT_TOKENS: int = 1536  # Max length of input text.
    MAX_TOTAL_TOKENS: int = 2048  # Max length of the generation (including input text).
    MAX_BATCH_TOTAL_TOKENS: int = 2048  # Limits the number of tokens that can be processed in parallel during the generation.

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 384
    EMBEDDING_MODEL_DEVICE: str = "cpu"

    def patch_localhost(self) -> None:
        self.QDRANT_DATABASE_HOST = "localhost"
        self.RABBITMQ_HOST = "localhost"


settings = AppSettings()
