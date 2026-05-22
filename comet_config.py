import os
from pathlib import Path

from comet_ml import API, start


_ENV_LOADED = False
_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_dotenv_once():
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    if _ENV_PATH.exists():
        for raw_line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    _ENV_LOADED = True


def _get_required_env(*names):
    _load_dotenv_once()
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined_names = " or ".join(names)
    raise RuntimeError(f"Missing {joined_names} in .env or environment.")


def get_comet_api_key():
    return _get_required_env("COMET_API_KEY")


def get_comet_workspace():
    return _get_required_env("COMET_WORKSPACE", "COMET_WORKSPACE_NAME")


def start_comet_experiment(project_name, **kwargs):
    kwargs.setdefault("api_key", get_comet_api_key())
    kwargs.setdefault("workspace", get_comet_workspace())
    return start(project_name=project_name, **kwargs)


def get_comet_api(**kwargs):
    kwargs.setdefault("api_key", get_comet_api_key())
    return API(**kwargs)
