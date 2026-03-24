"""
图像分类路由
"""
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import Optional
import uuid

from app.services.classification_service import classification_service
from app.core.config import settings

router = APIRouter()

# 允许的图片扩展名
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
# 最大上传文件大小 (50MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


@router.post("/train")
async def train_classification(
    train_path: str = Form(...),
    valid_path: Optional[str] = Form(None),
    epochs: int = Form(10),
    lr: float = Form(1e-3),
    batch_size: int = Form(2),  # 使用最小批次大小
    network_name: str = Form("resnet18"),  # 改名为 network_name 避免与 Pydantic 的 model_ 命名空间冲突
    resume_model: Optional[str] = Form(None),
    pretrained: bool = Form(False)  # 是否使用预训练模型
):
    """训练分类模型"""
    try:
        # 检查是否已有任务在进行
        from app.services.classification_service import training_progress
        if training_progress["classification"]["is_training"]:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "已有分类训练任务正在进行，请等待完成后再试"}
            )

        # 打印调试信息
        print("=" * 60)
        print("[BACKEND DEBUG] 收到的训练请求参数:")
        print(f"  network_name: {network_name} (type: {type(network_name).__name__})")
        print(f"  epochs: {epochs}")
        print(f"  lr: {lr}")
        print(f"  batch_size: {batch_size}")
        print(f"  pretrained: {pretrained}")
        print(f"  train_path: {train_path}")
        print(f"  valid_path: {valid_path}")
        print("=" * 60)

        # 验证路径
        if not Path(train_path).exists():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"训练数据路径不存在: {train_path}"}
            )

        # 如果提供了验证路径，也进行验证
        if valid_path and not Path(valid_path).exists():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"验证数据路径不存在: {valid_path}"}
            )

        # 训练任务（使用 run_in_threadpool 避免阻塞事件循环）
        from fastapi.concurrency import run_in_threadpool

        result = await run_in_threadpool(
            classification_service.train,
            train_path=train_path,
            valid_path=valid_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            model_name=network_name,
            resume_model=resume_model,
            pretrained=pretrained
        )

        # 深度清理确保可序列化
        return JSONResponse(content=deep_clean_for_json(result))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.post("/predict")
async def predict_classification(
    file: UploadFile = File(...),
    checkpoint_path: Optional[str] = Form(None)  # 改名为 checkpoint_path 避免与 Pydantic 的 model_ 命名空间冲突
):
    """图像分类预测"""
    try:
        # 训练期间可以进行预测，不阻止
        # 这样用户可以同时使用已训练好的模型进行推理，不影响新模型的训练

        # 验证文件名存在性
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "文件名不能为空"}
            )

        # 验证文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"不支持的文件类型: {file_ext}"}
            )

        # 确保上传目录存在
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 生成安全的文件名（防止路径遍历攻击）
        safe_filename = f"{uuid.uuid4().hex}{file_ext}"
        upload_path = settings.UPLOAD_DIR / safe_filename

        # 保存上传的文件（使用流式写入，支持大文件）
        chunk_size = 8192  # 8KB chunks
        total_size = 0

        with upload_path.open("wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)

                # 实时检查文件大小限制
                if total_size > MAX_UPLOAD_SIZE:
                    # 清理已写入的部分文件
                    buffer.close()
                    upload_path.unlink(missing_ok=True)
                    return JSONResponse(
                        status_code=413,
                        content={"success": False, "error": f"文件过大，最大允许 {MAX_UPLOAD_SIZE // 1024 // 1024} MB"}
                    )

                buffer.write(chunk)

        # 预测（使用 run_in_threadpool 避免阻塞事件循环）
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(
            classification_service.predict,
            image_path=str(upload_path),
            model_path=checkpoint_path  # 内部映射回 model_path
        )

        # 深度清理确保可序列化
        return JSONResponse(content=deep_clean_for_json(result))
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.post("/test")
async def test_classification(
    test_path: str = Form(...),
    checkpoint_path: Optional[str] = Form(None)  # 改名为 checkpoint_path 避免与 Pydantic 的 model_ 命名空间冲突
):
    """测试分类模型"""
    try:
        # 训练期间可以进行测试，不阻止
        # 这样用户可以对已有的模型进行测试评估

        # 验证路径
        if not Path(test_path).exists():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"测试数据路径不存在: {test_path}"}
            )

        # 启动测试任务（使用 run_in_threadpool 避免阻塞事件循环）
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(
            classification_service.test,
            test_path=test_path,
            model_path=checkpoint_path  # 内部映射回 model_path
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
    # 清理文件名，防止路径遍历攻击
    safe_filename = Path(filename).name

    # 验证文件名格式，防止路径遍历攻击
    if not safe_filename.startswith("classification_test_") or not safe_filename.endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "无效的文件名"}
        )

    # 安全地构建路径并验证
    result_path = (settings.RESULTS_DIR / safe_filename).resolve()
    if not result_path.is_relative_to(settings.RESULTS_DIR.resolve()):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": "非法访问"}
        )

    if result_path.exists():
        return FileResponse(result_path)
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "文件不存在"}
    )


@router.get("/models")
async def list_models():
    """列出所有分类模型"""
    # 确保模型目录存在
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = []
    for model_file in settings.MODELS_DIR.glob("classification_*.pkl"):
        # 只调用一次 stat() 获取所有文件信息
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
    # 验证文件名格式，防止删除非模型文件
    if not model_name.startswith("classification_") or not model_name.endswith(".pkl"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "无效的模型文件名"}
        )

    # 安全地构建路径并验证
    model_path = (settings.MODELS_DIR / model_name).resolve()
    if not model_path.is_relative_to(settings.MODELS_DIR.resolve()):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": "非法访问"}
        )

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
        from app.services.classification_service import training_progress
        progress = training_progress.get("classification", {
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
