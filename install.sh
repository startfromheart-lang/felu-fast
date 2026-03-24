#!/bin/bash

# FastAI AMD平台快速安装脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "FastAI AMD平台 - 快速安装"
echo "=========================================="

# 检查Python版本
echo ""
echo "1. 检查Python环境..."
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "✓ 找到 Python 3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✓ 使用 python3"
else
    echo "✗ 未找到Python，请先安装Python 3.12"
    exit 1
fi

# 创建虚拟环境
echo ""
echo "2. 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    $PYTHON_CMD -m venv venv
    echo "✓ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo ""
echo "3. 激活虚拟环境..."
source venv/bin/activate
echo "✓ 虚拟环境已激活"

# 升级pip
echo ""
echo "4. 升级pip和工具..."
pip install --upgrade pip setuptools wheel
echo "✓ pip已升级"

# 安装PyTorch ROCm版本
echo ""
echo "5. 安装PyTorch ROCm版本..."
echo "注意: PyTorch ROCm必须从官方源安装，这可能需要一些时间..."

# 尝试安装PyTorch 2.4.0 ROCm版本
if pip install torch==2.4.0+rocm7.1 torchvision==0.19.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1; then
    echo "✓ PyTorch 2.4.0 ROCm安装成功"
else
    echo "✗ PyTorch 2.4.0 安装失败，尝试安装最新可用版本..."
    if pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1; then
        echo "✓ PyTorch安装成功（最新版本）"
    else
        echo "✗ PyTorch安装失败，请检查网络连接或ROCm版本"
        exit 1
    fi
fi

# 安装其他依赖
echo ""
echo "6. 安装其他依赖..."
pip install -r requirements.txt
echo "✓ 其他依赖安装完成"

# 创建必要目录
echo ""
echo "7. 创建数据目录..."
mkdir -p data models results uploads
echo "✓ 数据目录创建完成"

# 验证安装
echo ""
echo "=========================================="
echo "8. 验证安装..."
echo "=========================================="

# 验证Python
PYTHON_VERSION=$(python --version 2>&1)
echo "Python: $PYTHON_VERSION"

# 验证PyTorch
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
echo "PyTorch: $TORCH_VERSION"

# 验证CUDA
CUDA_AVAILABLE=$(python -c "import torch; print('可用' if torch.cuda.is_available() else '不可用')" 2>/dev/null || echo "未安装")
echo "CUDA: $CUDA_AVAILABLE"

if [ "$CUDA_AVAILABLE" = "可用" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "GPU数量: $GPU_COUNT"
    echo "GPU名称: $GPU_NAME"
fi

# 验证FastAI
FASTAI_VERSION=$(python -c "import fastai; print(fastai.__version__)" 2>/dev/null || echo "未安装")
echo "FastAI: $FASTAI_VERSION"

# 总结
echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 验证GPU: python scripts/verify_gpu.py"
echo "  2. 准备数据: python scripts/prepare_data.py"
echo "  3. 启动服务: bash start.sh"
echo ""
echo "访问: http://localhost:8000"
echo ""
