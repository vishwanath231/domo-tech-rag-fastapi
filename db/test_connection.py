from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from mongodb import get_mongo_client

def test_connection():
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        print("✅ MongoDB connection successful!")
        return True

    except ServerSelectionTimeoutError as e:
        print("❌ MongoDB Timeout Error:", e)
        return False

    except ConnectionFailure as e:
        print("❌ MongoDB Connection Failed:", e)
        return False


if __name__ == "__main__":
    test_connection()
