import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# TTS model configurations
TTS_MODELS: Dict[str, str] = {
    "EN-US": "tts_models/en/ljspeech/tacotron2-DDC",
    "EN": "tts_models/en/ljspeech/tacotron2-DDC",
    "ES": "tts_models/es/css10/vits",
    "FR": "tts_models/fr/css10/vits",
    "ZH": "tts_models/zh-CN/baker/tacotron2-DDC",
    "JP": "tts_models/ja/kokoro/tacotron2-DDC"
}

# Whisper model configuration
WHISPER_MODEL_ID = "openai/whisper-small"

# Audio settings
SAMPLE_RATE = 16000

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german',
    'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese',
    'ja': 'japanese', 'ar': 'arabic', 'hi': 'hindi', 'nl': 'dutch',
    'pl': 'polish', 'tr': 'turkish'
}

# SSL Configuration
SSL_CERT_PATH = 'cert.pem'
SSL_KEY_PATH = 'key.pem'

# Server Configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
