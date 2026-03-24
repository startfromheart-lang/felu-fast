"""
图像分割路由
"""
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
from typing import Optional

from app.services.segmentation_service import segmentation_service
from app.core.config import settings

router = APIRouter()


@router.post("/train")
async def train_segmentation(
    background_tasks: BackgroundTasks,
    train_path: str = Form(...),
    epochs: int = Form(10),
    lr: float = Form(1e-3),
    batch_size: int = Form(1),  # 使用最小批次大小
    network_name: str = Form("resnet34"),  # 改名为 network_name 避免与 Pydantic 的 model_ 命名空间冲突
    resume_model: Optional[str] = Form(None),
    pretrained: bool = Form(False)  # 是否使用预训练模型
):
    """训练分割模型"""
    try:
        # 验证路径
        if not Path(train_path).exists():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"训练数据路径不存在: {train_path}"}
            )

        # 创建进度回调
        from fastapi.concurrency import run_in_threadpool

        progress_data = []

        def progress_callback(data):
            progress_data.append(data)

        # 启动训练任务（使用 run_in_threadpool 避免阻塞事件循环）
        result = await run_in_threadpool(
            segmentation_service.train,
            train_path=train_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            model_name=network_name,  # 内部映射回 model_name
            resume_model=resume_model,
            pretrained=pretrained,
            progress_callback=progress_callback
        )

        # 深度清理确保可序列化
        return JSONResponse(content=deep_clean_for_json(result))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.post("/predict")
async def predict_segmentation(
    file: UploadFile = File(...),
    checkpoint_path: Optional[str] = Form(None),  # 改名为 checkpoint_path 避免与 Pydantic 的 model_ 命名空间冲突
    return_overlay: bool = Form(True)
):
    """图像分割预测"""
    try:
        # 保存上传的文件
        upload_path = settings.UPLOAD_DIR / file.filename
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 预测（使用 run_in_threadpool 避免阻塞事件循环）
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(
            segmentation_service.predict,
            image_path=str(upload_path),
            model_path=checkpoint_path,  # 内部映射回 model_path
            return_overlay=return_overlay
        )

        # 深度清理确保可序列化
        return JSONResponse(content=deep_clean_for_json(result))
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.post("/test")
async def test_segmentation(
    background_tasks: BackgroundTasks,
    test_path: str = Form(...),
    checkpoint_path: Optional[str] = Form(None)  # 改名为 checkpoint_path 避免与 Pydantic 的 model_ 命名空间冲突
):
    """测试分割模型"""
    try:
        # 验证路径
        if not Path(test_path).exists():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"测试数据路径不存在: {test_path}"}
            )

        # 创建进度回调
        from fastapi.concurrency import run_in_threadpool

        progress_data = []

        def progress_callback(data):
            progress_data.append(data)

        # 启动测试任务（使用 run_in_threadpool 避免阻塞事件循环）
        result = await run_in_threadpool(
            segmentation_service.test,
            test_path=test_path,
            model_path=checkpoint_path,  # 内部映射回 model_path
            progress_callback=progress_callback
        )

        # 深度清理确保可序列化
        return JSONResponse(content=deep_clean_for_json(result))
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.get("/download-result/{filename}")
async def download_result(filename: str):
    """下载测试结果"""
    result_path = settings.RESULTS_DIR / filename
    if result_path.exists():
        return FileResponse(result_path)
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "文件不存在"}
    )


@router.get("/models")
async def list_models():
    """列出所有分割模型"""
    models = []
    for model_file in settings.MODELS_DIR.glob("segmentation_*.pkl"):
        stat = model_file.stat()
        # st_birthtime 在 Windows 上可用，Linux 上使用 st_ctime
        created_time = getattr(stat, 'st_birthtime', stat.st_ctime)
        models.append({
            "name": model_file.name,
            "path": str(model_file),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "created_time": float(created_time)  # 确保是 Python 原生 float
        })
    return {"models": sorted(models, key=lambda x: x["created_time"], reverse=True)}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """删除模型"""
    model_path = settings.MODELS_DIR / model_name
    if model_path.exists():
        model_path.unlink()
        return {"success": True, "message": "模型已删除"}
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "模型不存在"}
    )


def deep_clean_for_json(obj):
    """递归清理对象，确保可 JSON 序列化"""
    import numpy as np
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {key: deep_clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_clean_for_json(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


@router.get("/training-progress")
async def get_training_progress():
    """获取训练进度"""
    try:
        from app.services.segmentation_service import training_progress
        progress = training_progress.get("segmentation", {
            "is_training": False,
            "status": "idle",
            "message": ""
        })
        # 深度清理确保可序列化
        return deep_clean_for_json(progress)
    except Exception as e:
        return deep_clean_for_json({
            "is_training": False,
            "status": "error",
            "message": f"获取训练进度失败: {str(e)}",
            "error": str(e)
        })
