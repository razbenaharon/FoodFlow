import sys
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from httpx import ConnectError, ReadTimeout

def verify_qdrant_connection(host="localhost", port=6333, timeout=3.0):
    try:
        print(f"🔌 Checking Qdrant connection at http://{host}:{port} ...")
        client = QdrantClient(host=host, port=port, timeout=timeout)
        client.get_collections()
        print("✅ Qdrant is reachable and responding.")
        return client

    except ConnectError:
        print(f"❌ Qdrant is not reachable at {host}:{port}. Is the server running?")
        sys.exit(1)

    except ReadTimeout:
        print(f"⏳ Qdrant is too slow to respond (timeout after {timeout}s).")
        sys.exit(1)

    except UnexpectedResponse as e:
        print(f"⚠️ Qdrant responded with unexpected error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Unexpected error while connecting to Qdrant: {e}")
        sys.exit(1)


# verify_qdrant_connection()

from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

result = client.scroll(collection_name="food_recipes", limit=1, with_payload=True)
print(result)
