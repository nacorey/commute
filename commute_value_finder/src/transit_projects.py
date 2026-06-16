"""교통 호재(GTX·신설노선) 예정역 — 모델 피처가 아니라 Blue 후보 위 설명 레이어."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subway_access import haversine_km

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "transit_projects.json"


def load_projects(path=None):
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


def nearest_project_km(lat, lon, projects):
    """최근접 호재역까지 거리(km), 역명, 개통예정연도. 없으면 (inf, None, None)."""
    best = (float("inf"), None, None)
    for p in projects:
        d = haversine_km(lat, lon, p["lat"], p["lon"])
        if d < best[0]:
            best = (d, p["name"], p.get("open_year"))
    return best
