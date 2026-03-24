# 故障排除指南

本文档提供常见安装和运行问题的解决方案。

## 1. PyTorch ROCm安装问题

### 问题1: `No matching distribution found for torch==X.X.X+rocm7.1`

**原因**: PyTorch ROCm版本不在指定的镜像源中

**解决方案**:

```bash
# 必须从PyTorch官方源安装，不使用镜像源
pip install torch==2.4.0+rocm7.1 torchvision==0.19.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1

# 如果特定版本不可用，尝试安装最新版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

### 问题2: `pip install` 速度很慢

**解决方案**:

```bash
# PyTorch ROCm版本必须使用官方源，但可以设置超时
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 --timeout 1000

# 安装完成后，其他依赖可以使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题3: 安装时SSL证书错误

**解决方案**:

```bash
# 临时禁用SSL验证（不推荐，仅用于测试）
pip install --trusted-host download.pytorch.org torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

## 2. CUDA/GPU问题

### 问题4: `torch.cuda.is_available()` 返回 False

**检查步骤**:

```bash
# 1. 检查ROCm是否安装
/opt/rocm/bin/rocm-smi

# 2. 检查PyTorch是否支持ROCm
python -c "import torch; print(torch.__version__)"
# 应该看到类似: 2.4.0+rocm7.1

# 3. 检查环境变量
echo $LD_LIBRARY_PATH | grep rocm
echo $PATH | grep rocm
```

**解决方案**:

```bash
# 设置ROCm环境变量
source /etc/profile.d/rocm.sh

# 或临时设置
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 重新测试
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题5: 使用了集成显卡而非独立显卡

**检查**:

```bash
# 查看所有GPU
python scripts/verify_gpu.py
```

**解决方案**:

```python
# 在代码中显式选择独立显卡
import torch

# 列出所有GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# 选择显存最大的GPU（通常是独立显卡）
best_gpu_id = max(
    range(torch.cuda.device_count()),
    key=lambda i: torch.cuda.get_device_properties(i).total_memory
)
torch.cuda.set_device(best_gpu_id)
print(f"使用GPU {best_gpu_id}")
```

或在Web界面:
1. 访问 http://localhost:8000
2. 点击"查看所有GPU"
3. 找到独立显卡（显存较大）
4. 点击"选择"

## 3. 权限问题

### 问题6: ROCm权限错误

**错误信息**: `Permission denied` 或 `Access denied`

**解决方案**:

```bash
# 将用户添加到必要的组
sudo usermod -a -G render,video $USER

# 重新登录或执行
newgrp render
```

### 问题7: 无法创建/写入数据目录

**解决方案**:

```bash
# 手动创建目录并设置权限
mkdir -p data models results uploads
chmod -R 755 data models results uploads
```

## 4. 内存问题

### 问题8: 训练时显存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:

1. **减小批次大小**:

编辑 `.env` 文件:
```env
DEFAULT_BATCH_SIZE=8  # 从16改为8或更小
```

2. **减小图像尺寸**:

```python
# 在代码中调整图像大小
item_tfms=Resize(180)  # 从224改为180或更小
```

3. **使用梯度累积**:

```python
# 在训练代码中实现梯度累积
accumulation_steps = 4  # 模拟更大的batch size
```

4. **清理GPU缓存**:

```bash
# 在训练前或通过API清理
python -c "import torch; torch.cuda.empty_cache(); print('缓存已清理')"
```

## 5. Python版本问题

### 问题9: Python版本不兼容

**检查**:

```bash
python --version
# 应该是 Python 3.12.x
```

**解决方案**:

```bash
# 安装Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# 重新创建虚拟环境
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
```

## 6. 网络问题

### 问题10: 无法访问PyTorch官方源

**解决方案**:

```bash
# 方法1: 使用代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 方法2: 使用镜像（注意：ROCm版本仍需从官方源）
# 其他依赖可以使用镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

## 7. 服务启动问题

### 问题11: 端口8000已被占用

**错误**: `Address already in use`

**解决方案**:

```bash
# 方法1: 杀死占用端口的进程
sudo lsof -ti:8000 | xargs kill -9

# 方法2: 修改端口
# 编辑 .env 文件
PORT=8080  # 改为其他端口
```

### 问题12: 导入错误

**错误**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:

```bash
# 确认虚拟环境已激活
source venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt

# 如果是PyTorch问题
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

## 8. 获取帮助

### 收集诊断信息

```bash
# 运行完整诊断
bash scripts/diagnose.sh
```

### 手动收集信息

```bash
# 1. 系统信息
uname -a > diagnostics.txt
echo "" >> diagnostics.txt

# 2. Python信息
python --version >> diagnostics.txt
pip list >> diagnostics.txt
echo "" >> diagnostics.txt

# 3. ROCm信息
/opt/rocm/bin/rocm-smi >> diagnostics.txt
echo "" >> diagnostics.txt

# 4. PyTorch测试
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" >> diagnostics.txt 2>&1

# 将文件内容提交到Issue
cat diagnostics.txt
```

### 常用验证命令

```bash
# 验证ROCm
/opt/rocm/bin/rocm-smi
/opt/rocm/bin/rocminfo

# 验证Python
python --version
python -c "import sys; print(sys.version_info)"

# 验证PyTorch
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

# 验证GPU
python scripts/verify_gpu.py

# 验证安装
python scripts/verify_installation.py
```

## 9. 重新安装

### 完全清理并重新安装

```bash
# 停止服务
pkill -f "python main.py"

# 删除虚拟环境
rm -rf venv

# 重新创建
python3.12 -m venv venv
source venv/bin/activate

# 重新安装
bash install.sh
```

## 10. 联系支持

如果以上方案都无法解决问题：

1. 收集诊断信息（见上文）
2. 记录完整的错误信息和复现步骤
3. 提交到项目Issue或联系技术支持

---

**有用链接**:
- [PyTorch ROCm文档](https://pytorch.org/docs/stable/rocm.html)
- [ROCm文档](https://rocm.docs.amd.com/)
- [FastAI文档](https://docs.fast.ai/)
