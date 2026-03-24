"""
GPU工具模块 - 支持 NVIDIA CUDA 和 AMD ROCm
支持多GPU环境，优先使用独立显卡
"""
import torch
import os
import subprocess
import sys
from typing import List, Dict, Optional

# GPU检测结果缓存
_gpu_info_cache = None
_initialized = False

def is_windows() -> bool:
    """判断是否为Windows系统"""
    return sys.platform.startswith('win')

def check_nvidia_environment():
    """检查 NVIDIA CUDA 系统环境（Windows）"""
    print("检查 NVIDIA CUDA 系统环境...")

    # Windows上检查nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("  ✓ nvidia-smi 运行成功")
            # 提取驱动和CUDA版本信息
            lines = result.stdout.split('\n')
            for line in lines[:5]:  # 只看前5行的版本信息
                if 'NVIDIA-SMI' in line or 'CUDA Version' in line:
                    print(f"    {line.strip()}")
        else:
            print(f"  ⚠️  nvidia-smi 运行失败 (退出码: {result.returncode})")
    except FileNotFoundError:
        print(f"  ⚠️  未找到 nvidia-smi 命令（可能未安装NVIDIA驱动）")
    except Exception as e:
        print(f"  ⚠️  运行 nvidia-smi 时出错: {e}")

    # 检查CUDA环境变量
    cuda_env_vars = ['CUDA_PATH', 'CUDA_VERSION']
    for var in cuda_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✓ {var}={value}")
        else:
            print(f"    {var}= (未设置)")

    # 检查 PyTorch CUDA 模块信息
    try:
        print(f"  ✓ torch.cuda.is_available(): {torch.cuda.is_available()}")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"    torch.version.cuda: {torch.version.cuda}")
        if hasattr(torch.cuda, 'device_count'):
            print(f"    torch.cuda.device_count(): {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  ⚠️  获取 torch.cuda 信息失败: {e}")

    print()

def check_rocm_environment():
    """检查 ROCm 系统环境（Linux）"""
    print("检查 ROCm 系统环境...")

    # 检查 ROCm 工具
    rocm_tools = ['rocm-smi', 'rocminfo']
    for tool in rocm_tools:
        try:
            result = subprocess.run(
                ['which', tool] if not is_windows() else ['where', tool],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
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
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line or 'Card' in line or 'Device' in line:
                    print(f"    {line[:80]}")
    except Exception:
        pass

    # 检查环境变量
    rocm_env_vars = ['HIP_VISIBLE_DEVICES', 'HIP_PLATFORM', 'ROCM_PATH']
    for var in rocm_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✓ {var}={value}")
        else:
            print(f"    {var}= (未设置)")

    # 检查 PyTorch ROCm 模块信息
    try:
        print(f"  ✓ torch.cuda.is_available(): {torch.cuda.is_available()}")
        if hasattr(torch.version, 'hip'):
            print(f"    torch.version.hip: {torch.version.hip}")
        if hasattr(torch.cuda, 'device_count'):
            print(f"    torch.cuda.device_count(): {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  ⚠️  获取 torch 信息失败: {e}")

    print()

def get_gpu_backend() -> str:
    """获取当前GPU后端类型"""
    if hasattr(torch.version, 'hip') and torch.version.hip:
        return "ROCm (AMD)"
    elif hasattr(torch.version, 'cuda') and torch.version.cuda:
        return f"CUDA {torch.version.cuda} (NVIDIA)"
    return "Unknown"

def check_gpu_available(use_cache: bool = True, silent: bool = False) -> Dict:
    """检查GPU是否可用（支持NVIDIA CUDA和AMD ROCm）
    
    Args:
        use_cache: 是否使用缓存结果
        silent: 是否静默模式（不打印诊断信息）
    """
    global _gpu_info_cache
    
    # 使用缓存结果（如果可用）
    if use_cache and _gpu_info_cache is not None:
        return _gpu_info_cache
    
    # 打印 PyTorch 版本信息
    if not silent:
        print(f"PyTorch 版本: {torch.__version__}")
        backend = get_gpu_backend()
        if backend != "Unknown":
            print(f"计算后端: {backend}")
        else:
            print("⚠️  未检测到 GPU 后端支持")
        print()

        # 根据平台检查相应的环境
        if is_windows():
            check_nvidia_environment()
        else:
            check_rocm_environment()

    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        if not silent:
            print("⚠️  torch.cuda.is_available() 返回 False")
            print("   可能原因:")
            if is_windows():
                print("   1. 安装的 PyTorch 不是 CUDA 版本")
                print("   2. NVIDIA 驱动未正确安装")
                print("   3. CUDA 工具包版本不匹配")
                print()
                print("   建议操作:")
                print("   1. 检查安装命令: pip show torch")
                print("   2. 安装 CUDA 版本的 PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu128")
            else:
                print("   1. 安装的 PyTorch 不是 ROCm 版本")
                print("   2. ROCm 驱动未正确安装或未加载")
                print("   3. AMD GPU 未被系统识别")
            print()

        result = {
            "available": False,
            "device_count": 0,
            "devices": [],
            "selected_device": None,
            "device": "cpu",
            "device_id": None,
            "error": "torch.cuda.is_available() returned False",
            "torch_version": torch.__version__,
            "backend": backend if 'backend' in locals() else "Unknown"
        }
        _gpu_info_cache = result
        return result

    try:
        device_count = torch.cuda.device_count()

        if device_count == 0:
            if not silent:
                print("⚠️  torch.cuda.device_count() 返回 0")
                print("   CUDA 已加载，但未检测到设备")
            result = {
                "available": False,
                "device_count": 0,
                "devices": [],
                "selected_device": None,
                "device": "cpu",
                "device_id": None,
                "error": "No CUDA devices detected"
            }
            _gpu_info_cache = result
            return result

        gpus = []
        backend = get_gpu_backend()

        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3  # GB
                device_name = torch.cuda.get_device_name(i)

                gpus.append({
                    "id": i,
                    "name": device_name,
                    "memory_gb": round(total_memory, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "is_discrete": total_memory > 2.0,  # 独立显卡通常显存大于2GB
                    "backend": backend
                })
            except Exception as e:
                if not silent:
                    print(f"⚠️  获取 GPU {i} 信息失败: {e}")
                continue

        if not gpus:
            if not silent:
                print("⚠️  无法获取任何GPU的设备属性")
            result = {
                "available": False,
                "device_count": device_count,
                "devices": [],
                "selected_device": None,
                "device": "cpu",
                "device_id": None,
                "error": "Failed to get device properties"
            }
            _gpu_info_cache = result
            return result

        # 按显存排序，优先使用显存更大的显卡（通常是独立显卡）
        gpus_sorted = sorted(gpus, key=lambda x: x["memory_gb"], reverse=True)
        selected_gpu = gpus_sorted[0]

        result = {
            "available": True,
            "device_count": device_count,
            "devices": gpus_sorted,
            "selected_device": selected_gpu,
            "device": "cuda",
            "device_id": selected_gpu["id"],
            "backend": backend
        }
        _gpu_info_cache = result
        return result

    except Exception as e:
        if not silent:
            print(f"⚠️  GPU 检测过程中发生错误: {e}")
        result = {
            "available": False,
            "device_count": 0,
            "devices": [],
            "selected_device": None,
            "device": "cpu",
            "device_id": None,
            "error": str(e)
        }
        _gpu_info_cache = result
        return result


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
        os.environ["CUDA_VISIBLE_DEVICES"] = "all"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def select_discrete_gpu() -> Optional[int]:
    """选择独立显卡（显存最大的GPU）"""
    gpu_info = check_gpu_available()
    
    if not gpu_info["available"]:
        return None
    
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
    global _initialized
    
    gpu_info = check_gpu_available(silent=silent)

    if not gpu_info["available"]:
        if not silent:
            print("警告: 未检测到可用的GPU，将使用CPU")
        _initialized = True
        return torch.device("cpu")

    best_gpu_id = select_discrete_gpu()

    torch.cuda.set_device(best_gpu_id)

    device_info = gpu_info["selected_device"]
    if not silent:
        print(f"✓ 使用GPU: {device_info['name']}")
        print(f"✓ 显存: {device_info['memory_gb']} GB")
        print(f"✓ 设备ID: {best_gpu_id}")
        print(f"✓ 后端: {device_info['backend']}")
        print(f"✓ 设备类型: {'独立显卡' if device_info['is_discrete'] else '集成显卡'}")

    _initialized = True
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


def reset_gpu_cache():
    """重置GPU检测缓存"""
    global _gpu_info_cache, _initialized
    _gpu_info_cache = None
    _initialized = False
