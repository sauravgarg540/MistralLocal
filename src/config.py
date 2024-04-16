import os
from pydantic import field_validator
from pydantic_settings import BaseSettings
from .utils import get_cpu_or_gpu


LLM_MODELS = {
    "mistral_instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral": "mistralai/Mistral-7B-v0.1",
}


class Config(BaseSettings):
    MODEL: str = 'mistral_instruct'
    MAX_TOKENS: int = 2048
    DEPLOYMENT: str = 'serverless'
    HF_API_KEY: str = None
    USE_GPU: str = "True"
    DEVICE: str = get_cpu_or_gpu()

    @field_validator('MODEL')
    @classmethod
    def check_valid_model_name(cls, v: str) -> str:
        if v not in LLM_MODELS:
            raise ValueError(f"Model {v} not found in LLM_MODELS choose from {LLM_MODELS.keys()}")
        return LLM_MODELS[v]

    @field_validator('DEPLOYMENT')
    @classmethod
    def check_valid_deployment(cls, v: str) -> str:
        if v not in ['serverless', 'local']:
            raise ValueError("DEPLOYMENT must be either 'serverless' or 'local'")
        return v

    @field_validator('HF_API_KEY')
    @classmethod
    def check_valid_deployment(cls, v: str, values, **kwargs) -> str:
        if values.data['DEPLOYMENT'] == 'serverless' and v is None:
            raise ValueError("HF_API_KEY must be set for serverless deployment")
        return v

    @field_validator('USE_GPU')
    @classmethod
    def check_valid_use_gpu(cls, v: str, values, **kwargs) -> bool:
        if v.lower() in ('false', 'f', '0', 'off', 'n', 'no'):
            v = False
        elif v.lower() in ('true', 't', '1', 'on', 'y', 'yes'):
            v = True
        return v

    @field_validator('DEVICE')
    @classmethod
    def check_valid_device(cls, v: str, values, **kwargs) -> str:
        if values.data['USE_GPU']:
            if v == 'cpu':
                print('No GPU available, using CPU instead.')
        return v

config = Config()
