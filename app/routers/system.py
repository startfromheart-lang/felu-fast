"""
系统路由 - GPU状态检查等
"""
from fastapi import APIRouter
from app.core.gpu_utils import (
    check_gpu_available,
    get_memory_info,
    get_all_gpu_memory_info,
    clear_gpu_cache,
    get_gpu_selection_info,
    test_gpu_computation,
    initialize_best_gpu
)
from app.core.config import settings

router = APIRouter()


# 延迟初始化，避免模块导入时立即检测GPU
_device_initialized = False
selected_device = None


def ensure_device_initialized():
    """确保设备已初始化（不打印警告）"""
    global _device_initialized, selected_device
    if not _device_initialized:
        selected_device = initialize_best_gpu(silent=True)
        _device_initialized = True
    return selected_device


@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 确保设备已初始化（静默，不打印警告）
        ensure_device_initialized()
        gpu_info = check_gpu_available()
        selection_info = get_gpu_selection_info()

        return {
            "gpu": gpu_info,
            "selection": selection_info,
            "platform": "GPU" if gpu_info["available"] else "CPU"
        }
    except Exception as e:
        import traceback
        return {
            "error": f"获取系统状态失败: {str(e)}",
            "gpu": {
                "available": False,
                "device_count": 0,
                "devices": [],
                "selected_device": None,
                "device": "cpu",
                "device_id": None,
                "error": str(e)
            },
            "selection": {
                "using_gpu": False,
                "device": "cpu",
                "message": f"GPU检测失败: {str(e)}"
            },
            "platform": "CPU Only"
        }


@router.get("/config")
async def get_config():
    """获取系统配置信息"""
    return {
        "data_dir": str(settings.DATA_DIR),
        "models_dir": str(settings.MODELS_DIR),
        "results_dir": str(settings.RESULTS_DIR),
        "upload_dir": str(settings.UPLOAD_DIR),
        "default_paths": {
            "classification": {
                "train": str(settings.DATA_DIR / "color" / "train"),
                "valid": str(settings.DATA_DIR / "color" / "valid"),
                "test": str(settings.DATA_DIR / "color" / "test")
            },
            "segmentation": {
                "train": str(settings.DATA_DIR / "segment" / "train"),
                "test": str(settings.DATA_DIR / "segment" / "test")
            }
        }
    }


@router.get("/gpus")
async def list_gpus():
    """列出所有GPU及其详细信息"""
    # 确保设备已初始化（静默，不打印警告）
    ensure_device_initialized()
    gpu_info = check_gpu_available()
    
    if not gpu_info["available"]:
        return {
            "available": False,
            "gpus": [],
            "message": "未检测到可用的GPU"
        }
    
    # 获取所有GPU的内存信息
    all_memory_info = get_all_gpu_memory_info()
    
    return {
        "available": True,
        "total_count": gpu_info["device_count"],
        "selected_device_id": gpu_info["device_id"],
        "gpus": [
            {
                "id": gpu["id"],
                "name": gpu["name"],
                "memory_gb": gpu["memory_gb"],
                "compute_capability": gpu["compute_capability"],
                "is_discrete": gpu["is_discrete"],
                "gpu_type": "独立显卡" if gpu["is_discrete"] else "集成显卡",
                "memory_info": next((m for m in all_memory_info if m["device_id"] == gpu["id"]), None)
            }
            for gpu in gpu_info["devices"]
        ]
    }


@router.get("/gpu/{device_id}/memory")
async def get_gpu_memory(device_id: int):
    """获取指定GPU的内存信息"""
    memory_info = get_memory_info(device_id)
    
    if memory_info is None:
        return {
            "success": False,
            "message": f"无法获取GPU {device_id} 的内存信息"
        }
    
    return {
        "success": True,
        "memory": memory_info
    }


@router.get("/gpu/{device_id}/test")
async def test_gpu(device_id: int):
    """测试指定GPU的计算能力"""
    test_result = test_gpu_computation(device_id)
    return test_result


@router.post("/select-gpu/{device_id}")
async def select_gpu(device_id: int):
    """选择指定GPU"""
    import torch
    
    gpu_info = check_gpu_available()
    
    if not gpu_info["available"]:
        return {
            "success": False,
            "message": "未检测到可用的GPU"
        }
    
    if device_id >= gpu_info["device_count"]:
        return {
            "success": False,
            "message": f"GPU ID {device_id} 不存在"
        }
    
    try:
        torch.cuda.set_device(device_id)
        global selected_device, _device_initialized
        selected_device = torch.device(f"cuda:{device_id}")
        _device_initialized = True
        
        gpu_name = torch.cuda.get_device_name(device_id)
        return {
            "success": True,
            "message": f"已选择GPU {device_id}: {gpu_name}",
            "device_id": device_id,
            "device_name": gpu_name
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"选择GPU失败: {str(e)}"
        }


@router.post("/clear-cache")
async def clear_cache(device_id: int = None):
    """清理GPU缓存"""
    success = clear_gpu_cache(device_id)
    return {
        "success": success,
        "message": f"GPU {device_id if device_id is not None else '所有'} 缓存已清理" if success else "GPU不可用"
    }