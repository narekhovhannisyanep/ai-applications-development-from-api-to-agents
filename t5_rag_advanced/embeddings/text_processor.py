import json
from enum import StrEnum

import psycopg2
import psycopg2.errorcodes
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
        )

    def _truncate_table(self):
        """Truncate the vectors table."""
        connection = None
        try:
            connection = self._get_connection()
            with connection.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors;")
                connection.commit()
                print("Table 'vecttors' truncated successfully.")
        except Exception as e:
            print(f"Error tuncating table: {e}")
        finally:
            if connection is not None:
                connection.close()

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        document_name_without_path = document_name.split("/")[-1]
        try:
            connection = self._get_connection()
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                insert_query = """
                INSERT INTO vectors (document_name, text, embedding)
                VALUES (%s, %s, %s::vector)
                RETURNING *;
                """
                embedding_data = (document_name_without_path, text, embedding)
                cursor.execute(insert_query, embedding_data)
                connection.commit()

        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            print(f"Database Error during save: {e}")
        except Exception as e:
            print(f"An unexpected error occured during save: {e}")
        finally:
            if connection is not None:
                connection.close()

    def process_text_file(
        self,
        file_name: str,
        chunk_size: int,
        overlap: int,
        dimensions: int,
        should_truncate: bool,
    ):
        """
        Reads a file, splits the content into chunks, generates vectors, and saves to database.
        """
        if should_truncate:
            self._truncate_table()

        with open(file_name, "r", encoding="utf-8") as f:
            chunks = chunk_text(f.read(), chunk_size, overlap)

            embeddings = self.embeddings_client.get_embeddings(
                chunks, dimensions, False
            )

            for text, embedding in zip(chunks, embeddings.values()):
                self._save_chunk(file_name, text, embedding)

    def search(
        self,
        search_mode: SearchMode,
        user_request: str,
        top_k: int,
        max_distance: float,
        dimensions: int,
    ):
        raw_embeddings = self.embeddings_client.get_embeddings(user_request, dimensions)
        user_vector = next(iter(raw_embeddings.values()))
        operator = "<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"
        search_query = f"""
        SELECT text, embedding {operator} %s::vector AS distance
        FROM vectors
        WHERE embedding {operator} %s::vector < %s
        ORDER BY distance ASC
        LIMIT %s;
        """

        try:
            connection = self._get_connection()
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    search_query, (user_vector, user_vector, max_distance, top_k)
                )
                select_results = cursor.fetchall()
                return select_results
        except psycopg2.OperationalError as e:
            print(f"DB Operation error: {e}")
            raise e from e
        except Exception as e:
            print(f"An unexpected error occured: {e}")
            raise e from e
        finally:
            if connection is not None:
                connection.close()
