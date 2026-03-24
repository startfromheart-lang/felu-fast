"""
FastAI AMD图像分类与分割平台主入口
"""
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from app.routers import classification, segmentation, system
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("FastAI AMD平台启动中...")
    print(f"服务地址: http://{settings.HOST}:{settings.PORT}")
    print()

    # 检测并显示GPU信息
    from app.core.gpu_utils import check_gpu_available

    print("-" * 60)
    print("检测计算设备...")
    print("-" * 60)

    gpu_info = check_gpu_available()

    if gpu_info["available"]:
        selected_gpu = gpu_info.get("selected_device", {})
        print()
        print("=" * 60)
        print("✓ 检测到可用GPU:")
        print(f"  设备名称: {selected_gpu.get('name', 'Unknown')}")
        print(f"  显存大小: {selected_gpu.get('memory_gb', 'Unknown')} GB")
        print(f"  设备类型: {'独立显卡' if selected_gpu.get('is_discrete') else '集成显卡'}")
        print(f"  设备ID: {gpu_info.get('device_id', 'Unknown')}")
        if selected_gpu.get('backend'):
            print(f"  后端: {selected_gpu.get('backend', 'Unknown')}")
        if gpu_info.get("device_count", 1) > 1:
            print(f"  检测到 {gpu_info['device_count']} 个GPU设备")
        print("=" * 60)
    else:
        # GPU检测失败，但已在 check_gpu_available 中打印了详细原因
        error = gpu_info.get("error", "Unknown error")
        print()
        print("=" * 60)
        print("⚠️  未检测到可用的GPU，将使用CPU进行计算")
        print(f"  错误原因: {error}")
        print("  提示: 请检查 ROCm 驱动和 PyTorch ROCm 版本是否正确安装")
        print("=" * 60)

    print()

    yield
    # 关闭时清理
    print("FastAI AMD平台关闭...")


app = FastAPI(
    title="FastAI AMD图像分类与分割平台",
    description="基于FastAI和AMD ROCm的深度学习平台",
    version="1.0.0",
    lifespan=lifespan
)

# 挂载路由
app.include_router(system.router, prefix="/api/system", tags=["系统"])
app.include_router(classification.router, prefix="/api/classification", tags=["图像分类"])
app.include_router(segmentation.router, prefix="/api/segmentation", tags=["图像分割"])

# 挂载静态文件（禁用缓存以便开发调试）
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 挂载结果目录作为静态文件
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")


@app.get("/")
async def root():
    """主页"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
