from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_overall_counts_keys():
    response = client.get("/overall_counts")
    assert response.status_code == 200
    body = response.json()
    for key in [
        "customers",
        "orders",
        "products",
        "employees",
        "offices",
        "payments",
        "orderdetails",
        "productlines",
    ]:
        assert key in body
        assert isinstance(body[key], int)
