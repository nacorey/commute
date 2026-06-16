"""공통 유틸리티 함수"""

import os
from pathlib import Path
from dotenv import load_dotenv


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환"""
    return Path(__file__).resolve().parent.parent


def load_env(required_keys: list[str] | None = None) -> dict[str, str]:
    """
    .env 파일에서 API 키를 로드하고 검증한다.

    Args:
        required_keys: 검증할 키 목록. None이면 전체 키 검증.

    Returns:
        API 키 딕셔너리

    Raises:
        EnvironmentError: 필수 키가 없거나 템플릿 값 그대로일 때
    """
    env_path = get_project_root() / ".env"
    load_dotenv(env_path)

    all_keys = ["MOLIT_API_KEY", "KAKAO_API_KEY", "OPENAI_API_KEY"]
    check_keys = required_keys if required_keys else all_keys

    env = {}
    for key in all_keys:
        env[key] = os.getenv(key)

    missing = [
        k for k in check_keys
        if not env.get(k) or env[k].startswith("여기에")
    ]
    if missing:
        raise EnvironmentError(
            f"다음 API 키가 설정되지 않았습니다: {', '.join(missing)}\n"
            f".env 파일을 확인하세요: {env_path}"
        )

    return env
