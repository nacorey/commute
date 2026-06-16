from src.transit_projects import load_projects, nearest_project_km


def test_load_projects_real_file():
    projects = load_projects()
    assert len(projects) >= 10
    assert all("lat" in p and "line" in p for p in projects)


def test_nearest_project_km():
    projects = [
        {"name": "GTX-A 삼성", "line": "GTX-A", "lat": 37.5088, "lon": 127.0631, "open_year": 2028},
        {"name": "GTX-C 창동", "line": "GTX-C", "lat": 37.6530, "lon": 127.0477, "open_year": 2028},
    ]
    dist, name, year = nearest_project_km(37.5100, 127.0640, projects)
    assert name == "GTX-A 삼성"
    assert dist < 1.0
    assert year == 2028


def test_nearest_project_empty():
    dist, name, year = nearest_project_km(37.5, 127.0, [])
    assert name is None
