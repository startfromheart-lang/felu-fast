"""
应用配置文件
"""
import os
from pydantic_settings import BaseSettings
from pathlib import Path


# 获取项目根目录（无论在哪个目录运行服务）
# 使用当前文件的绝对路径向上查找项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """应用配置"""

    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = True

    # 数据目录（使用项目根目录下的绝对路径）
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    # 训练配置
    DEFAULT_EPOCHS: int = 10
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_LR: float = 1e-3

    # 图像配置
    MAX_IMAGE_SIZE: int = 224
    ALLOWED_IMAGE_TYPES: list = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # ROCm配置
    FORCE_GPU: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# 打印目录信息用于调试
print(f"✓ 项目根目录: {BASE_DIR}")
print(f"✓ 数据目录: {settings.DATA_DIR}")
print(f"✓ 模型目录: {settings.MODELS_DIR}")
print(f"✓ 结果目录: {settings.RESULTS_DIR}")
print(f"✓ 上传目录: {settings.UPLOAD_DIR}")

# 创建必要的目录
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
