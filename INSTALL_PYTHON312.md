# Python 3.12 安装指南

本文档提供在Ubuntu 22.04上安装和配置Python 3.12的详细步骤。

## 1. 安装Python 3.12

### 1.1 添加deadsnakes PPA仓库

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
```

### 1.2 安装Python 3.12

```bash
# 安装Python 3.12及相关工具
sudo apt install python3.12 python3.12-venv python3.12-dev python3.12-distutils -y

# 验证安装
python3.12 --version
# 预期输出: Python 3.12.x
```

### 1.3 安装pip

```bash
# 下载并安装pip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.12 get-pip.py

# 验证pip
python3.12 -m pip --version
```

### 1.4 设置默认Python（可选）

```bash
# 如果需要将python3.12设为默认python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

# 选择默认版本
sudo update-alternatives --config python3
```

## 2. 创建Python 3.12虚拟环境

```bash
# 在项目目录中创建虚拟环境
python3.12 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip到最新版本
pip install --upgrade pip setuptools wheel
```

## 3. 安装项目依赖

### 3.1 安装PyTorch ROCm版本

```bash
# 方案1: 安装PyTorch 2.4.0 ROCm版本（稳定版本，推荐）
pip install torch==2.4.0+rocm7.1 torchvision==0.19.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1

# 方案2: 尝试安装更新的版本（如果方案1不可用）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

**注意**: PyTorch ROCm版本必须从官方PyTorch源安装，不支持pip镜像源（如清华源）。

### 3.2 安装其他依赖

```bash
# 安装其他项目依赖（不包含PyTorch）
pip install -r requirements.txt
```

## 4. 验证安装

### 4.1 验证Python版本

```bash
# 创建测试脚本
cat > test_installation.py << 'EOF'
import sys
import torch
import fastai

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"FastAI version: {fastai.__version__}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU计算
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"GPU computation test: PASSED (result sum: {z.sum().item():.4f})")
else:
    print("WARNING: CUDA is not available. Check ROCm installation.")
EOF

# 运行测试
python test_installation.py
```

### 4.3 验证关键依赖

```bash
# 快速验证所有关键依赖
python -c "import fastapi, uvicorn, pydantic, PIL, cv2, numpy, pandas, sklearn, matplotlib; print('所有依赖安装成功！')"
```

## 5. 依赖版本说明

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.x | 主解释器 |
| PyTorch | 2.6.0+rocm7.1 | 深度学习框架（ROCm版本） |
| Torchvision | 0.21.0+rocm7.1 | 计算机视觉库（ROCm版本） |
| FastAI | 2.7.18 | 高级深度学习API |
| FastAPI | 0.115.0 | Web框架 |
| Uvicorn | 0.32.0 | ASGI服务器 |
| NumPy | 1.26.4 | 数值计算库 |
| Pandas | 2.2.3 | 数据分析库 |
| Pillow | 11.0.0 | 图像处理库 |
| OpenCV | 4.10.0.84 | 计算机视觉库 |

## 6. 常见问题

### 问题1：pip install失败

```bash
# 解决方案：升级pip和相关工具
pip install --upgrade pip setuptools wheel

# 如果还是失败，尝试使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2：PyTorch安装找不到ROCm版本

```bash
# 检查可用的PyTorch版本
pip index versions torch

# 如果2.6.0不可用，尝试最新的可用版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

### 问题3：CUDA不可用

```bash
# 检查ROCm安装
/opt/rocm/bin/rocminfo
/opt/rocm/bin/rocm-smi

# 确保环境变量已设置
echo $PATH | grep rocm
echo $LD_LIBRARY_PATH | grep rocm

# 如果没有，执行：
source /etc/profile.d/rocm.sh
```

### 问题4：虚拟环境激活失败

```bash
# 确保使用Python 3.12创建虚拟环境
python3.12 -m venv --clear venv
source venv/bin/activate
```

## 7. 性能优化建议

### 7.1 使用pip缓存加速

```bash
# 设置pip缓存目录
mkdir -p ~/.cache/pip

# 下载时使用缓存
pip install --cache-dir ~/.cache/pip -r requirements.txt
```

### 7.2 使用国内镜像源（可选）

```bash
# 创建pip配置文件
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

### 7.3 PyTorch性能优化

```bash
# 设置MIOpen缓存目录
mkdir -p ~/.cache/miopen

# 添加到环境变量
export MIOPEN_USER_DB_PATH=$HOME/.cache/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$HOME/.cache/miopen
```

## 8. 下一步

安装完成后：

1. **准备数据集**:
   ```bash
   python scripts/prepare_data.py
   ```

2. **启动服务**:
   ```bash
   bash start.sh
   ```

3. **访问Web界面**:
   浏览器打开 `http://localhost:8000`

## 9. 卸载和清理

### 卸载虚拟环境

```bash
# 先 deactivate 虚拟环境
deactivate

# 删除虚拟环境目录
rm -rf venv
```

### 卸载Python 3.12（谨慎操作）

```bash
# 卸载Python 3.12
sudo apt remove python3.12 python3.12-venv python3.12-dev python3.12-distutils -y

# 移除PPA仓库
sudo add-apt-repository --remove ppa:deadsnakes/ppa -y
```

## 10. 参考资源

- Python 3.12官方文档: https://docs.python.org/3.12/
- PyTorch文档: https://pytorch.org/docs/stable/index.html
- FastAI文档: https://docs.fast.ai/
- ROCm文档: https://rocm.docs.amd.com/
