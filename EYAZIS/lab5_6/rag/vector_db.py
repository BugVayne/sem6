import os
import json
import logging
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self, db_path="music_db.json"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.db = []

        try:
            logger.info(f"Initializing VectorDB with database at: {db_path}")

            if not os.path.exists(db_path):
                error_msg = f"{db_path} not found"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            with open(db_path) as f:
                self.db = json.load(f)
            logger.info(f"Successfully loaded database with {len(self.db)} items")

            if not isinstance(self.db, list):
                error_msg = "Database should be a list of records"
                logger.error(error_msg)
                raise ValueError(error_msg)

            self.build_index()

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            self.index = None

    def build_index(self):
        if not self.db:
            error_msg = "No data to index"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Building FAISS index...")

        # Combine text, tags, and category for embedding
        data_for_embedding = [
            f"{item['text']} {' '.join(item['tags'])} {item['category']}" for item in self.db
        ]

        embeddings = self.model.encode(
            data_for_embedding,
            convert_to_numpy=True
        ).astype('float32')

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        logger.info(f"Built index with {len(self.db)} items (embedding dimension: {embeddings.shape[1]})")

    def search(self, query, k=3):
        if self.index is None:
            error_msg = "Index not initialized - cannot perform search"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Processing search query: '{query}' (k={k})")

        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        logger.info(f"Search completed. Top {k} results found.")

        results = [(self.db[i]) for idx, i in enumerate(indices[0]) if distances[0][idx] < 1]

        for result in results:
            logger.info(f"\tresult:\t{result}")
        return results


def test_vectordb():

    db = VectorDB("music_db.json")

    queries = [
        "Gojira band progressive metal album"
    ]

    for query in queries:
        print(f"\nSearching for: '{query}'")

        try:
            results = db.search(query, k=3)  # Get top 3 matches

            for result in results:
                print(f"→ {result['text']}, {result['tags']}, {result['category']}")

        except Exception as e:
            print(f"Error during search: {e}")

    logger.info("VectorDB test completed.")


if __name__ == "__main__":
    test_vectordb()