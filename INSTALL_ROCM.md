# ROCm安装详细指南

## 1. 安装ROCm驱动

### 1.1 添加ROCm仓库

```bash
# 添加ROCm GPG密钥
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# 添加ROCm仓库（Ubuntu 22.04）
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.1/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
```

### 1.2 安装ROCm

```bash
# 更新软件包列表
sudo apt update

# 安装ROCm核心包
sudo apt install rocm-dev rocm-libs rocm-utils rocm-opencl-dev

# 验证安装
/opt/rocm/bin/rocminfo
```

### 1.3 配置环境变量

```bash
# 添加环境变量
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' | sudo tee -a /etc/profile.d/rocm.sh

# 激活环境变量
source /etc/profile.d/rocm.sh
```

### 1.4 配置用户权限

```bash
# 将当前用户添加到render和video组
sudo usermod -a -G render,video $USER

# 重新登录以使权限生效
# 或执行以下命令立即生效（推荐使用新的登录会话）
newgrp render
```

## 2. 验证ROCm安装

```bash
# 查看GPU状态
/opt/rocm/bin/rocm-smi

# 预期输出示例：
# ============================================== ROCm System Management Interface =============================================
# ==================================================== Concise Info =========================================================
# GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK    Fan    Perf  PwrCap  VRAM%  GPU%
# 0    46.0C           25.0W   1600MHz  945MHz  0.00%  auto  200.0W   1%   0%
# ==========================================================================================================================

# 查看GPU详细信息
/opt/rocm/bin/rocminfo
```

## 3. 安装PyTorch ROCm版本

### 3.1 创建虚拟环境

```bash
# 创建Python 3.12虚拟环境
python3.12 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### 3.2 安装PyTorch

```bash
# 安装PyTorch 2.4.0 ROCm版本（稳定版）
pip install torch==2.4.0+rocm7.1 torchvision==0.19.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1
```

### 3.3 安装其他依赖

```bash
# 安装项目依赖（不包含PyTorch）
pip install -r requirements.txt
```

**注意**: requirements.txt中不包含PyTorch版本限制，因为PyTorch ROCm版本必须从官方源安装。

### 3.4 验证PyTorch GPU支持

```bash
# 创建测试脚本
cat > test_gpu.py << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU计算
    x = torch.rand(3, 3).cuda()
    y = torch.rand(3, 3).cuda()
    z = x + y
    print(f"GPU computation test: {z.sum().item()}")
else:
    print("CUDA is not available. Please check ROCm installation.")
EOF

# 运行测试
python test_gpu.py
```

## 3. 安装FastAI和其他依赖

```bash
# 安装项目依赖（包含FastAI）
pip install -r requirements.txt
```

## 5. 常见问题排查

### 问题1：找不到rocm命令

```bash
# 解决方案：手动添加到PATH
export PATH=/opt/rocm/bin:$PATH
```

### 问题2：权限被拒绝

```bash
# 解决方案：检查用户组
groups $USER

# 确保用户在render和video组中
sudo usermod -a -G render,video $USER

# 重新登录
```

### 问题3：torch.cuda.is_available()返回False

```bash
# 检查ROCm安装
/opt/rocm/bin/rocminfo

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 确保安装的是ROCm版本（包含+rocm）
```

### 问题4：显存不足

```bash
# 在训练时减小批次大小
# 修改 .env 文件中的 DEFAULT_BATCH_SIZE 为更小的值，如 8 或 4
```

## 6. 性能优化

### 6.1 启用MIOpen缓存

```bash
# 创建MIOpen缓存目录
mkdir -p ~/.cache/miopen

# 设置环境变量
export MIOPEN_USER_DB_PATH=$HOME/.cache/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$HOME/.cache/miopen
```

### 6.2 监控GPU使用情况

```bash
# 实时监控
watch -n 1 /opt/rocm/bin/rocm-smi

# 训练时监控
/opt/rocm/bin/rocm-smi --showpids
```

## 7. 卸载ROCm（如需要）

```bash
# 停止ROCm服务
sudo /opt/rocm/bin/rocminfo

# 卸载ROCm包
sudo apt remove rocm-* rocm-dev rocm-libs rocm-utils

# 清理配置
sudo rm -rf /etc/profile.d/rocm.sh
sudo rm -rf /opt/rocm
```

## 8. 参考资源

- ROCm官方文档: https://rocm.docs.amd.com/
- PyTorch ROCm: https://pytorch.org/
- FastAI文档: https://docs.fast.ai/
