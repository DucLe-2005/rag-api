from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Batch, Distance, VectorParams

import core.logger_utils as logger_utils
import os
import dotenv

dotenv.load_dotenv()

logger = logger_utils.get_logger(__name__)


class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __init__(self) -> None:
        if self._instance is None:
                self._instance = QdrantClient(
                    url=os.getenv("QDRANT_CLOUD_URL"),
                api_key=os.getenv("QDRANT_APIKEY")
                )

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def create_non_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config={"dummy": VectorParams(size=1, distance=Distance.COSINE)},
        )

    def create_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_SIZE, distance=Distance.COSINE
            ),
        )

    def write_data(self, collection_name: str, points: Batch):
        try:
            self._instance.upsert(collection_name=collection_name, points=points)
        except Exception:
            logger.exception("An error occurred while inserting data.")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self):
        if self._instance:
            self._instance.close()
            logger.info("Connected to database has been closed.")