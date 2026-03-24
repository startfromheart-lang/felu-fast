#!/bin/bash

# FastAI AMD平台启动脚本

echo "=========================================="
echo "FastAI AMD图像分类与分割平台"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
else
    echo "警告: 虚拟环境不存在，使用系统Python"
fi

# 检查ROCm
echo ""
echo "检查ROCm状态..."
if [ -f "/opt/rocm/bin/rocminfo" ]; then
    echo "ROCm已安装"
    /opt/rocm/bin/rocm-smi
else
    echo "警告: ROCm未找到，将使用CPU运行"
fi

# 检查PyTorch GPU支持
echo ""
echo "检查PyTorch GPU支持..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 询问是否进行GPU验证
echo ""
read -p "是否进行GPU性能验证？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "运行GPU验证..."
    python scripts/verify_gpu.py
    echo ""
fi

# 创建必要目录
echo ""
echo "创建数据目录..."
mkdir -p data models results uploads

# 启动服务
echo ""
echo "=========================================="
echo "启动FastAPI服务..."
echo "服务地址: http://localhost:8000"
echo "=========================================="
echo ""

# 启动时显示GPU信息
echo "正在初始化GPU..."
python main.py
