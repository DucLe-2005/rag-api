import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/api/health")
    print("\n=== Testing Health Check ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_collection_types():
    """Test the collection types endpoint."""
    response = requests.get(f"{BASE_URL}/api/collection-types")
    print("\n=== Testing Collection Types ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert "collection_types" in response.json()

def test_query():
    """Test the query endpoint with different scenarios."""
    # Test 1: Basic query with date range
    print("\n=== Testing Query with Date Range ===")
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "query": "What were the major market events in the last quarter?",
            "top_k": 5,
            "expand_n_query": 2,
            "keep_top_k": 3
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert "context" in response.json()
    assert isinstance(response.json()["context"], list)
    assert len(response.json()["context"]) > 0

    # Test 2: Query with specific time period
    print("\n=== Testing Query with Specific Time Period ===")
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "query": "Show me the latest earnings reports from this month",
            "top_k": 5,
            "expand_n_query": 2,
            "keep_top_k": 3
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert "context" in response.json()
    assert isinstance(response.json()["context"], list)
    assert len(response.json()["context"]) > 0

    # Test 3: Query without date range
    print("\n=== Testing Query without Date Range ===")
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "query": "What are the key financial metrics for tech companies?",
            "top_k": 5,
            "expand_n_query": 2,
            "keep_top_k": 3
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert "context" in response.json()
    assert isinstance(response.json()["context"], list)
    assert len(response.json()["context"]) > 0

def main():
    """Run all tests."""
    try:
        test_health()
        test_collection_types()
        test_query()
        print("\n=== All tests completed successfully! ===")
    except AssertionError as e:
        print(f"\n=== Test failed: {str(e)} ===")
    except requests.exceptions.ConnectionError:
        print("\n=== Error: Could not connect to the API. Make sure the server is running. ===")
    except Exception as e:
        print(f"\n=== Unexpected error: {str(e)} ===")

if __name__ == "__main__":
    main() 