"""pytest 공통 픽스처."""
import sys
from pathlib import Path

# src 임포트를 위해 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


@pytest.fixture
def sample_transactions():
    """정제·모델 테스트용 소형 거래 데이터프레임."""
    return pd.DataFrame(
        {
            "구": ["강남구", "강남구", "노원구", "노원구", "노원구"],
            "법정동": ["삼성동", "삼성동", "상계동", "상계동", "상계동"],
            "아파트명": ["A아파트", "A아파트", "B아파트", "B아파트", "C아파트"],
            "전용면적": [84.0, 59.0, 84.0, 84.0, 49.0],
            "거래금액": [200000, 150000, 60000, 62000, 40000],
            "층": [10, 3, 5, 15, 2],
            "건축년도": [2010, 2010, 1995, 1995, 1988],
            "년": [2026, 2026, 2026, 2025, 2025],
            "월": [3, 1, 5, 12, 6],
        }
    )
