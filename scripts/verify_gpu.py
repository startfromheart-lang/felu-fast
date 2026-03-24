#!/usr/bin/env python3
"""
GPU验证脚本 - 检测和验证AMD独立显卡
确保使用独立显卡而非集成显卡
"""
import sys
import torch
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_rocm_installation():
    """检查ROCm安装"""
    print_header("1. ROCm系统检查")
    
    # 检查rocm-smi
    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ ROCm工具已安装")
            print("\nGPU列表:")
            print(result.stdout)
            return True
        else:
            print("✗ ROCm工具执行失败")
            return False
    except FileNotFoundError:
        print("✗ ROCm未安装，请先安装ROCm驱动")
        return False
    except Exception as e:
        print(f"✗ 检查ROCm失败: {e}")
        return False


def check_pytorch_rocm():
    """检查PyTorch ROCm支持"""
    print_header("2. PyTorch ROCm检查")
    
    try:
        version = torch.__version__
        has_rocm = "rocm" in version.lower()
        
        print(f"PyTorch版本: {version}")
        print(f"ROCm支持: {'是' if has_rocm else '否'}")
        
        if not has_rocm:
            print("✗ PyTorch未使用ROCm版本，请重新安装ROCm版本的PyTorch")
            return False
        
        return True
    except Exception as e:
        print(f"✗ PyTorch检查失败: {e}")
        return False


def check_all_gpus():
    """检查所有GPU"""
    print_header("3. GPU设备检查")
    
    try:
        if not torch.cuda.is_available():
            print("✗ CUDA不可用")
            return None
        
        device_count = torch.cuda.device_count()
        print(f"\n检测到 {device_count} 个GPU设备:\n")
        
        gpus = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            name = torch.cuda.get_device_name(i)
            total_memory = props.total_memory / 1024**3  # GB
            
            gpu_info = {
                "id": i,
                "name": name,
                "memory_gb": round(total_memory, 2),
                "is_discrete": total_memory > 2.0  # 通常独立显卡显存大于2GB
            }
            gpus.append(gpu_info)
            
            gpu_type = "独立显卡 (推荐)" if gpu_info["is_discrete"] else "集成显卡"
            print(f"GPU {i}: {name}")
            print(f"  显存: {gpu_info['memory_gb']} GB")
            print(f"  类型: {gpu_type}")
            print(f"  计算能力: {props.major}.{props.minor}")
            print()
        
        # 推荐最佳GPU
        if gpus:
            best_gpu = max(gpus, key=lambda x: x["memory_gb"])
            print(f"推荐使用: GPU {best_gpu['id']} ({best_gpu['name']})")
            print(f"原因: 显存最大 ({best_gpu['memory_gb']} GB)，类型: {'独立显卡' if best_gpu['is_discrete'] else '集成显卡'}")
        
        return gpus
    except Exception as e:
        print(f"✗ GPU检查失败: {e}")
        return None


def test_gpu_computation(gpu_id):
    """测试指定GPU的计算能力"""
    print_header(f"4. GPU {gpu_id} 性能测试")
    
    try:
        # 设置GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"使用设备: {torch.cuda.get_device_name(gpu_id)}")
        print()
        
        # 测试1: 张量创建
        print("测试1: 创建张量...")
        size = 2000
        x = torch.randn(size, size, device=device)
        print(f"  ✓ 创建 {size}x{size} 张量成功")
        
        # 测试2: 矩阵乘法
        print("测试2: 矩阵乘法...")
        import time
        y = torch.randn(size, size, device=device)
        
        # 预热
        for _ in range(3):
            _ = x @ y
        torch.cuda.synchronize()
        
        # 实际测试
        start = time.time()
        for _ in range(5):
            z = x @ y
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_time = (elapsed / 5) * 1000  # 转换为毫秒
        gflops = (2 * size**3) / (avg_time / 1000) / 1e9
        
        print(f"  ✓ 平均时间: {avg_time:.2f} ms")
        print(f"  ✓ 计算性能: ~{gflops:.1f} GFLOPS")
        
        # 测试3: 内存分配
        print("测试3: 内存分配...")
        allocated_before = torch.cuda.memory_allocated(gpu_id) / 1024**3
        large_tensor = torch.randn(500, 1000, 1000, device=device)
        allocated_after = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"  ✓ 分配内存: {(allocated_after - allocated_before):.2f} GB")
        
        # 测试4: Deep Learning模拟
        print("测试4: 简单神经网络模拟...")
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 100)
        ).to(device)
        
        input_tensor = torch.randn(100, size, device=device)
        
        # 预热
        for _ in range(3):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        
        # 实际测试
        start = time.time()
        for _ in range(10):
            output = model(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  ✓ 前向传播平均时间: {(elapsed/10)*1000:.2f} ms (10次)")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU测试失败: {e}")
        return False


def check_gpu_memory(gpu_id):
    """检查GPU内存状态"""
    print_header(f"5. GPU {gpu_id} 内存状态")
    
    try:
        torch.cuda.set_device(gpu_id)
        
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        free = total - reserved
        
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"总显存: {total:.2f} GB")
        print(f"已分配: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"已保留: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"可用: {free:.2f} GB ({free/total*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"✗ 内存检查失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  AMD GPU验证工具 - 检测独立显卡并验证性能")
    print("=" * 70)
    
    # 检查ROCm
    rocm_ok = check_rocm_installation()
    if not rocm_ok:
        print("\n⚠️  请先安装ROCm驱动，参考 INSTALL_ROCM.md")
        return 1
    
    # 检查PyTorch ROCm
    pytorch_ok = check_pytorch_rocm()
    if not pytorch_ok:
        print("\n⚠️  请安装ROCm版本的PyTorch:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1")
        return 1
    
    # 检查所有GPU
    gpus = check_all_gpus()
    if not gpus:
        print("\n⚠️  未检测到可用的GPU")
        return 1
    
    # 选择最佳GPU（独立显卡）
    best_gpu = max(gpus, key=lambda x: x["memory_gb"])
    print(f"\n{'='*70}")
    print(f"  将使用GPU {best_gpu['id']}: {best_gpu['name']}")
    print(f"  类型: {'独立显卡' if best_gpu['is_discrete'] else '集成显卡'}")
    print(f"{'='*70}\n")
    
    # 测试GPU计算
    computation_ok = test_gpu_computation(best_gpu['id'])
    
    # 检查内存
    memory_ok = check_gpu_memory(best_gpu['id'])
    
    # 总结
    print_header("验证总结")
    
    results = {
        "ROCm安装": rocm_ok,
        "PyTorch ROCm": pytorch_ok,
        "GPU检测": len(gpus) > 0,
        "GPU计算": computation_ok,
        "GPU内存": memory_ok
    }
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"{icon} {name:<20} {'通过' if status else '失败'}")
    
    all_ok = all(results.values())
    
    print(f"\n总计: {sum(results.values())}/{len(results)} 项通过")
    
    if all_ok:
        print("\n🎉 所有检查通过！")
        print(f"✓ 系统已正确配置使用 {'独立显卡' if best_gpu['is_discrete'] else '集成显卡'}")
        print("\n下一步:")
        print("  1. 启动服务: bash start.sh")
        print("  2. 访问界面: http://localhost:8000")
        print("  3. 在主页点击'查看所有GPU'确认使用的是独立显卡")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请查看上述详情。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
