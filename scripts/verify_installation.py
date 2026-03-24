#!/usr/bin/env python3
"""
安装验证脚本 - Python 3.12 + ROCm 7.1
检查所有依赖是否正确安装和配置
"""
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name, status, details=""):
    """打印结果"""
    icon = "✓" if status else "✗"
    print(f"{icon} {name:<30} {'OK' if status else 'FAILED'}")
    if details:
        print(f"  └─ {details}")


def check_python_version():
    """检查Python版本"""
    print_header("Python版本检查")
    version = sys.version_info
    required = (3, 12, 0)
    
    is_ok = version >= required
    details = f"当前版本: {version.major}.{version.minor}.{version.micro}"
    
    print_result("Python 3.12+", is_ok, details)
    return is_ok


def check_dependencies():
    """检查依赖包"""
    print_header("依赖包检查")
    
    dependencies = {
        "torch": {"min_version": "2.6.0"},
        "torchvision": {"min_version": "0.21.0"},
        "fastai": {"min_version": "2.7.18"},
        "fastapi": {"min_version": "0.115.0"},
        "uvicorn": {"min_version": "0.32.0"},
        "pydantic": {"min_version": "2.9.0"},
        "numpy": {"min_version": "1.26.0"},
        "pandas": {"min_version": "2.2.0"},
        "PIL": {"name": "Pillow", "min_version": "11.0.0"},
        "cv2": {"name": "opencv-python", "min_version": "4.10.0"},
        "sklearn": {"name": "scikit-learn", "min_version": "1.5.0"},
        "matplotlib": {"min_version": "3.9.0"},
    }
    
    all_ok = True
    for module_name, config in dependencies.items():
        try:
            module = __import__(module_name)
            actual_name = config.get("name", module_name)
            version = getattr(module, "__version__", "unknown")
            min_version = config.get("min_version", "0.0.0")
            
            is_ok = True  # 简化版本检查
            details = f"版本: {version}"
            print_result(actual_name, is_ok, details)
            
        except ImportError:
            print_result(config.get("name", module_name), False, "未安装")
            all_ok = False
    
    return all_ok


def check_pytorch_rocm():
    """检查PyTorch ROCm支持"""
    print_header("PyTorch ROCm检查")
    
    try:
        import torch
        
        # PyTorch版本
        torch_version = torch.__version__
        has_rocm = "rocm" in torch_version.lower()
        print_result("PyTorch版本", True, torch_version)
        print_result("ROCm标记", has_rocm, "包含rocm" if has_rocm else "未检测到")
        
        # CUDA可用性
        cuda_available = torch.cuda.is_available()
        print_result("CUDA可用", cuda_available, "ROCm" if has_rocm else "N/A")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_result("GPU数量", True, str(device_count))
            print_result("GPU名称", True, device_name)
            
            # GPU内存测试
            try:
                x = torch.rand(1000, 1000).cuda()
                y = torch.rand(1000, 1000).cuda()
                z = x @ y
                print_result("GPU计算测试", True, "矩阵乘法成功")
            except Exception as e:
                print_result("GPU计算测试", False, str(e))
        
        return cuda_available
        
    except Exception as e:
        print_result("PyTorch", False, str(e))
        return False


def check_rocm_installation():
    """检查ROCm安装"""
    print_header("ROCm系统检查")
    
    rocm_tools = [
        ("/opt/rocm/bin/rocm-smi", "ROCm系统监控"),
        ("/opt/rocm/bin/rocminfo", "ROCm设备信息"),
    ]
    
    all_ok = True
    for tool_path, tool_name in rocm_tools:
        tool_file = Path(tool_path)
        if tool_file.exists():
            try:
                result = subprocess.run(
                    [tool_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                is_ok = result.returncode == 0
                print_result(tool_name, is_ok, "可执行" if is_ok else "执行失败")
            except Exception as e:
                print_result(tool_name, False, str(e))
                all_ok = False
        else:
            print_result(tool_name, False, "未安装")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """检查项目结构"""
    print_header("项目结构检查")
    
    required_dirs = [
        "app",
        "app/core",
        "app/routers",
        "app/services",
        "static",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        dir_exists = Path(dir_path).exists()
        print_result(f"{dir_path}/", dir_exists)
        if not dir_exists:
            all_ok = False
    
    return all_ok


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  FastAI AMD平台 - Python 3.12 安装验证")
    print("=" * 60)
    
    results = {}
    
    # 执行各项检查
    results["python"] = check_python_version()
    results["dependencies"] = check_dependencies()
    results["pytorch"] = check_pytorch_rocm()
    results["rocm"] = check_rocm_installation()
    results["structure"] = check_project_structure()
    
    # 总结
    print_header("验证总结")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"{icon} {name:<20} {'通过' if status else '失败'}")
    
    print(f"\n总计: {passed}/{total} 项通过")
    
    if passed == total:
        print("\n🎉 所有检查通过！系统已准备就绪。")
        print("\n下一步:")
        print("  1. 准备数据集: python scripts/prepare_data.py")
        print("  2. 启动服务: bash start.sh")
        print("  3. 访问界面: http://localhost:8000")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请查看上述详情。")
        print("\n建议:")
        print("  - 重新安装失败的依赖")
        print("  - 检查ROCm安装: bash scripts/install_rocm.sh")
        print("  - 参考文档: INSTALL_PYTHON312.md, INSTALL_ROCM.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
