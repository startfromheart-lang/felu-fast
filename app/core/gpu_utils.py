"""
GPU工具模块 - AMD ROCm适配
支持多GPU环境，优先使用独立显卡
"""
import torch
import os
import subprocess
import sys
from fastai.vision.all import *
from typing import List, Dict, Optional


def check_rocm_environment():
    """检查 ROCm 系统环境"""
    print("检查 ROCm 系统环境...")

    # 检查 ROCm 工具
    rocm_tools = ['rocm-smi', 'opticon', 'rocminfo']
    found_tools = []

    for tool in rocm_tools:
        try:
            result = subprocess.run(
                ['which', tool],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                found_tools.append(tool)
                print(f"  ✓ 找到工具: {tool}")
        except:
            pass

    # 尝试直接运行 rocm-smi
    try:
        result = subprocess.run(
            ['rocm-smi'],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            print(f"  ✓ rocm-smi 运行成功")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line or 'Card' in line or 'Device' in line:
                    print(f"    {line[:80]}")
        else:
            print(f"  ⚠️  rocm-smi 运行失败 (退出码: {result.returncode})")
    except FileNotFoundError:
        print(f"  ⚠️  未找到 rocm-smi 命令")
    except Exception as e:
        print(f"  ⚠️  运行 rocm-smi 时出错: {e}")

    # 检查环境变量
    rocm_env_vars = ['HIP_VISIBLE_DEVICES', 'HIP_PLATFORM', 'ROCM_PATH']
    for var in rocm_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✓ {var}={value}")
        else:
            print(f"    {var}= (未设置)")

    # 检查 PyTorch CUDA 模块信息
    try:
        print(f"  ✓ torch.cuda.is_available(): {torch.cuda.is_available()}")
        if hasattr(torch.version, 'cuda'):
            print(f"    torch.version.cuda: {torch.version.cuda}")
        if hasattr(torch.version, 'hip'):
            print(f"    torch.version.hip: {torch.version.hip}")
        if hasattr(torch.cuda, 'device_count'):
            print(f"    torch.cuda.device_count(): {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  ⚠️  获取 torch.cuda 信息失败: {e}")

    print()


def check_gpu_available() -> Dict:
    """检查GPU是否可用（ROCm环境）
    参照 gpu_compute.py 的检测逻辑，增加更详细的错误诊断
    """
    # 打印 PyTorch 和 ROCm 版本信息
    print(f"PyTorch 版本: {torch.__version__}")
    if hasattr(torch.version, 'hip'):
        print(f"ROCm 版本: {torch.version.hip}")
    else:
        print("⚠️  未检测到 ROCm 后端（torch.version.hip 不存在）")
    print()

    # 检查系统环境
    check_rocm_environment()

    # 检查 torch 是否可用
    if not torch.cuda.is_available():
        print("⚠️  torch.cuda.is_available() 返回 False")
        print("   可能原因:")
        print("   1. 安装的 PyTorch 不是 ROCm 版本")
        print("   2. ROCm 驱动未正确安装或未加载")
        print("   3. AMD GPU 未被系统识别")
        print("   4. 环境变量未正确设置")
        print()
        print("   建议操作:")
        print("   1. 检查安装命令: pip show torch")
        print("   2. 运行: rocm-smi 或 /opt/rocm/bin/rocm-smi")
        print("   3. 检查环境变量: echo $HIP_VISIBLE_DEVICES")
        print()

        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "selected_device": None,
            "device": "cpu",
            "device_id": None,
            "error": "torch.cuda.is_available() returned False",
            "torch_version": torch.__version__,
            "has_hip": hasattr(torch.version, 'hip'),
            "hip_version": torch.version.hip if hasattr(torch.version, 'hip') else None
        }

    try:
        device_count = torch.cuda.device_count()

        if device_count == 0:
            print("⚠️  torch.cuda.device_count() 返回 0")
            print("   CUDA/ROCm 已加载，但未检测到设备")
            return {
                "available": False,
                "device_count": 0,
                "devices": [],
                "selected_device": None,
                "device": "cpu",
                "device_id": None,
                "error": "No CUDA devices detected"
            }

        gpus = []

        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3  # GB
                device_name = torch.cuda.get_device_name(i)

                # 检测是否为 ROCm 后端
                backend = "ROCm (AMD)" if hasattr(torch.version, 'hip') and torch.version.hip else "CUDA (NVIDIA)"

                gpus.append({
                    "id": i,
                    "name": device_name,
                    "memory_gb": round(total_memory, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "is_discrete": total_memory > 2.0,  # 独立显卡通常显存大于2GB
                    "backend": backend
                })
            except Exception as e:
                print(f"⚠️  获取 GPU {i} 信息失败: {e}")
                continue

        if not gpus:
            print("⚠️  无法获取任何GPU的设备属性")
            return {
                "available": False,
                "device_count": device_count,
                "devices": [],
                "selected_device": None,
                "device": "cpu",
                "device_id": None,
                "error": "Failed to get device properties"
            }

        # 按显存排序，优先使用显存更大的显卡（通常是独立显卡）
        gpus_sorted = sorted(gpus, key=lambda x: x["memory_gb"], reverse=True)
        selected_gpu = gpus_sorted[0]

        return {
            "available": True,
            "device_count": device_count,
            "devices": gpus_sorted,
            "selected_device": selected_gpu,
            "device": "cuda",
            "device_id": selected_gpu["id"]
        }

    except Exception as e:
        print(f"⚠️  GPU 检测过程中发生错误: {e}")
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "selected_device": None,
            "device": "cpu",
            "device_id": None,
            "error": str(e)
        }


def get_default_device() -> torch.device:
    """获取默认设备（优先使用独立显卡）"""
    gpu_info = check_gpu_available()
    if gpu_info["available"]:
        device_id = gpu_info["device_id"]
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def set_visible_devices(device_ids: List[int] = None):
    """设置可见的GPU设备"""
    if device_ids is None:
        # 如果未指定，使用所有GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "all"
    else:
        # 指定特定的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def select_discrete_gpu() -> Optional[int]:
    """选择独立显卡（显存最大的GPU）"""
    gpu_info = check_gpu_available()
    
    if not gpu_info["available"]:
        return None
    
    # 寻找显存最大的GPU（通常是独立显卡）
    discrete_gpu = max(
        gpu_info["devices"],
        key=lambda x: x["memory_gb"]
    )
    
    return discrete_gpu["id"]


def get_memory_info(device_id: int = 0) -> Optional[Dict]:
    """获取指定GPU的内存信息"""
    try:
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
            return {
                "device_id": device_id,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2),
                "device_name": torch.cuda.get_device_name(device_id)
            }
    except Exception as e:
        print(f"⚠️  获取 GPU {device_id} 内存信息失败: {e}")

    return None


def get_all_gpu_memory_info() -> List[Dict]:
    """获取所有GPU的内存信息"""
    gpu_info = check_gpu_available()
    
    if not gpu_info["available"]:
        return []
    
    memory_infos = []
    for gpu in gpu_info["devices"]:
        info = get_memory_info(gpu["id"])
        if info:
            memory_infos.append(info)
    
    return memory_infos


def clear_gpu_cache(device_id: int = None):
    """清理GPU缓存"""
    if torch.cuda.is_available():
        if device_id is not None:
            torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        return True
    return False


def test_gpu_computation(device_id: int = 0) -> Dict:
    """测试GPU计算能力"""
    try:
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)

            # 测试矩阵运算
            size = 1000
            x = torch.randn(size, size).cuda()
            y = torch.randn(size, size).cuda()

            import time
            start = time.time()
            z = x @ y
            result = z.sum().item()
            elapsed = time.time() - start

            return {
                "success": True,
                "device_id": device_id,
                "device_name": torch.cuda.get_device_name(device_id),
                "computation_time_ms": round(elapsed * 1000, 2),
                "result": round(result, 4)
            }
        else:
            return {
                "success": False,
                "device_id": device_id,
                "error": f"GPU {device_id} not available or out of range"
            }
    except Exception as e:
        print(f"⚠️  GPU {device_id} 计算测试失败: {e}")
        return {
            "success": False,
            "device_id": device_id,
            "error": str(e)
        }


def initialize_best_gpu(silent: bool = False):
    """初始化最佳GPU（独立显卡优先）

    Args:
        silent: 是否静默模式（不打印信息）
    """
    gpu_info = check_gpu_available()

    if not gpu_info["available"]:
        if not silent:
            print("警告: 未检测到可用的GPU，将使用CPU")
        return torch.device("cpu")

    # 选择显存最大的GPU（独立显卡）
    best_gpu_id = select_discrete_gpu()

    # 设置当前设备
    torch.cuda.set_device(best_gpu_id)

    device_info = gpu_info["selected_device"]
    if not silent:
        print(f"✓ 使用GPU: {device_info['name']}")
        print(f"✓ 显存: {device_info['memory_gb']} GB")
        print(f"✓ 设备ID: {best_gpu_id}")
        print(f"✓ 设备类型: {'独立显卡' if device_info['is_discrete'] else '集成显卡'}")

    return torch.device(f"cuda:{best_gpu_id}")


def get_gpu_selection_info() -> Dict:
    """获取GPU选择信息"""
    gpu_info = check_gpu_available()
    
    if not gpu_info["available"]:
        return {
            "using_gpu": False,
            "device": "cpu",
            "message": "未检测到可用的GPU"
        }
    
    selected = gpu_info["selected_device"]
    
    return {
        "using_gpu": True,
        "device": f"cuda:{gpu_info['device_id']}",
        "device_id": gpu_info["device_id"],
        "device_name": selected["name"],
        "memory_gb": selected["memory_gb"],
        "is_discrete": selected["is_discrete"],
        "gpu_type": "独立显卡" if selected["is_discrete"] else "集成显卡",
        "all_gpus": gpu_info["devices"],
        "message": f"使用{selected['is_discrete'] and '独立' or '集成'}显卡: {selected['name']}"
    }
