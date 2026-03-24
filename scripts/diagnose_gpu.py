#!/usr/bin/env python3
"""
GPU 诊断脚本
用于检测和诊断 ROCm/PyTorch GPU 环境
"""
import torch
import subprocess
import os
import sys


def check_rocm_tools():
    """检查 ROCm 工具"""
    print("=" * 70)
    print("1. 检查 ROCm 系统工具")
    print("=" * 70)

    rocm_tools = ['rocm-smi', 'rocminfo', 'opticon']
    for tool in rocm_tools:
        try:
            result = subprocess.run(
                ['which', tool],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print(f"✓ {tool}: {result.stdout.strip()}")
            else:
                print(f"✗ {tool}: 未找到")
        except Exception as e:
            print(f"✗ {tool}: 检测失败 - {e}")

    # 尝试运行 rocm-smi 获取详细信息
    print("\n运行 rocm-smi:")
    try:
        result = subprocess.run(
            ['rocm-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"✗ rocm-smi 运行失败 (退出码: {result.returncode})")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
    except FileNotFoundError:
        print("✗ 未找到 rocm-smi 命令")
    except Exception as e:
        print(f"✗ 运行 rocm-smi 失败: {e}")

    print()


def check_environment_variables():
    """检查环境变量"""
    print("=" * 70)
    print("2. 检查环境变量")
    print("=" * 70)

    env_vars = {
        'HIP_VISIBLE_DEVICES': '控制可见的 GPU 设备',
        'HIP_PLATFORM': 'HIP 平台类型',
        'ROCM_PATH': 'ROCm 安装路径',
        'LD_LIBRARY_PATH': '库搜索路径',
        'CUDA_VISIBLE_DEVICES': 'CUDA 可见设备（不适用于 ROCm）'
    }

    for var, desc in env_vars.items():
        value = os.environ.get(var, '')
        if value:
            print(f"✓ {var}={value}")
        else:
            print(f"  {var}=(未设置) - {desc}")

    print()


def check_pytorch_info():
    """检查 PyTorch 信息"""
    print("=" * 70)
    print("3. 检查 PyTorch 信息")
    print("=" * 70)

    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")

    # 检查后端
    if hasattr(torch.version, 'hip'):
        print(f"✓ ROCm 后端: {torch.version.hip}")
    else:
        print("✗ 未检测到 ROCm 后端")

    if hasattr(torch.version, 'cuda'):
        print(f"  CUDA 版本: {torch.version.cuda}")

    # 检查 GPU 可用性
    print(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ GPU 可用")
        print(f"  设备数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                name = torch.cuda.get_device_name(i)
                memory_gb = props.total_memory / 1024**3
                print(f"\n  GPU {i}:")
                print(f"    名称: {name}")
                print(f"    显存: {memory_gb:.2f} GB")
                print(f"    计算能力: {props.major}.{props.minor}")
            except Exception as e:
                print(f"  GPU {i}: 获取信息失败 - {e}")
    else:
        print("✗ GPU 不可用")

        # 尝试获取更多信息
        try:
            if hasattr(torch.cuda, 'device_count'):
                count = torch.cuda.device_count()
                print(f"  torch.cuda.device_count() = {count}")
        except Exception as e:
            print(f"  调用 torch.cuda.device_count() 失败: {e}")

    print()


def test_gpu_computation():
    """测试 GPU 计算"""
    print("=" * 70)
    print("4. 测试 GPU 计算")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("✗ GPU 不可用，跳过计算测试")
        return

    print("运行矩阵乘法测试...")

    try:
        device = torch.device("cuda:0")
        size = 1000

        # 创建张量
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)

        # 预热
        for _ in range(3):
            _ = x @ y
            torch.cuda.synchronize()

        # 测试
        import time
        start = time.time()
        z = x @ y
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"✓ 计算成功!")
        print(f"  矩阵大小: {size}x{size}")
        print(f"  耗时: {elapsed*1000:.2f} ms")
        print(f"  结果形状: {z.shape}")

    except Exception as e:
        print(f"✗ 计算失败: {e}")

    print()


def check_installation():
    """检查安装情况"""
    print("=" * 70)
    print("5. 检查安装情况")
    print("=" * 70)

    # 检查 pip 安装信息
    packages = ['torch', 'fastai', 'torchvision']
    for pkg in packages:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', pkg],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                version = next((line for line in lines if line.startswith('Version:')), '')
                location = next((line for line in lines if line.startswith('Location:')), '')
                print(f"✓ {pkg}")
                print(f"  {version}")
                if location:
                    print(f"  {location}")
            else:
                print(f"✗ {pkg}: 未安装")
        except Exception as e:
            print(f"✗ {pkg}: 检测失败 - {e}")

    print()


def main():
    """主函数"""
    print()
    print("=" * 70)
    print("GPU 诊断工具 - ROCm/PyTorch 环境")
    print("=" * 70)
    print()

    check_rocm_tools()
    check_environment_variables()
    check_pytorch_info()
    check_installation()
    test_gpu_computation()

    print("=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
