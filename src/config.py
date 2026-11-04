import os
from dotenv import load_dotenv
from sentinelhub import SHConfig

# Загружаем переменные из .env
load_dotenv()

def get_sentinel_config() -> SHConfig:
    config = SHConfig()
    config.instance_id = os.getenv("SH_INSTANCE_ID")
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    return config

# Базовые пути проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

