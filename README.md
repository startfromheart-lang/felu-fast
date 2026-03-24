# FastAI AMD图像分类与分割平台

基于FastAI和AMD ROCm的深度学习图像处理平台，支持图像分类与分割任务的训练、测试和推理服务。

ref: https://www.biosino.org/TonguExpert/index

## 功能特性

### 图像分类
- 模型训练：支持自定义数据集训练，可配置训练参数
- 模型测试：批量测试并生成详细报告
- 在线推理：上传图片实时分类，显示各类别概率分布

### 图像分割
- 模型训练：基于U-Net架构的语义分割模型训练
- 模型测试：批量分割测试，计算Dice和IoU指标
- 在线分割：上传图片实时分割，生成mask和叠加图

### 系统特性
- AMD ROCm GPU加速支持
- **多GPU智能选择**：自动识别并使用独立显卡（显存最大的GPU）
- 实时训练进度监控
- Web界面友好操作
- RESTful API接口
- 异步任务处理
- GPU性能监控和测试

## 环境要求

- 操作系统：Ubuntu 22.04.5
- GPU：AMD显卡（支持ROCm）
- Python：3.12
- ROCm：7.1+

> **注意**: 详细的Python 3.12和ROCm安装指南请参考 [INSTALL_PYTHON312.md](INSTALL_PYTHON312.md) 和 [INSTALL_ROCM.md](INSTALL_ROCM.md)

## 安装步骤

### 1. 安装ROCm驱动

```bash
# 添加ROCm仓库
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.1/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 更新并安装ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs rocm-utils

# 配置环境变量
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' | sudo tee -a /etc/profile.d/rocm.sh

# 激活环境
source /etc/profile.d/rocm.sh

# 将用户添加到render组
sudo usermod -a -G render,video $USER
```

### 2. 安装Python 3.12

如果系统尚未安装Python 3.12，请参考 [INSTALL_PYTHON312.md](INSTALL_PYTHON312.md) 进行安装。

### 3. 安装Python依赖

**方法1: 使用快速安装脚本（推荐）**

```bash
bash install.sh
```

**方法2: 手动安装**

```bash
# 创建虚拟环境（Python 3.12）
python3.12 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel

# 安装PyTorch ROCm版本（必须从官方源安装）
pip install torch==2.4.0+rocm7.1 torchvision==0.19.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1

# 安装其他依赖
pip install -r requirements.txt
```

**重要说明**:
- PyTorch ROCm版本**必须**从PyTorch官方源安装，不支持pip镜像源（如清华源）
- `requirements.txt`中不包含PyTorch版本限制，需单独安装ROCm版本
- 如果`2.4.0+rocm7.1`不可用，可尝试: `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1`
pip install -r requirements.txt
```

### 3. 安装FastAI

```bash
pip install fastai
```

### 5. 验证安装

```bash
# 验证所有依赖
python scripts/verify_installation.py

# 验证GPU并检查独立显卡选择
python scripts/verify_gpu.py
```

**GPU选择说明**：
- 系统会自动选择显存最大的GPU（通常是独立显卡）
- 如果同时有集成显卡和独立显卡，会优先使用独立显卡
- 可在Web界面查看和管理所有GPU

## 项目结构

```
aihoo-fast-service/
├── app/
│   ├── core/               # 核心模块
│   │   ├── config.py       # 配置文件
│   │   └── gpu_utils.py    # GPU工具
│   ├── routers/            # 路由模块
│   │   ├── system.py       # 系统路由
│   │   ├── classification.py  # 分类路由
│   │   └── segmentation.py     # 分割路由
│   └── services/           # 服务模块
│       ├── classification_service.py  # 分类服务
│       └── segmentation_service.py     # 分割服务
├── static/                 # 前端文件
│   ├── index.html          # 主页面
│   └── app.js              # 前端脚本
├── data/                   # 数据目录（自动创建）
├── models/                 # 模型目录（自动创建）
├── results/                # 结果目录（自动创建）
├── uploads/                # 上传目录（自动创建）
├── main.py                 # 主入口
├── requirements.txt        # 依赖文件
└── README.md              # 说明文档
```

## 使用方法

### 启动服务

```bash
source venv/bin/activate
python main.py
```

服务将在 `http://localhost:8000` 启动

### 访问Web界面

浏览器打开 `http://localhost:8000`

## 数据集格式

### 图像分类数据集

标准目录结构：
```
dataset/
├── train/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   └── dog/
│       ├── dog1.jpg
│       └── dog2.jpg
└── valid/
    ├── cat/
    │   └── cat3.jpg
    └── dog/
        └── dog3.jpg
```

### 图像分割数据集

标准目录结构：
```
segmentation_dataset/
├── images/           # 原始图像
│   ├── img1.jpg
│   └── img2.jpg
├── labels/           # 标签mask
│   ├── img1.png
│   └── img2.png
└── codes.txt         # 类别名称列表
```

`codes.txt` 格式示例：
```
background
class1
class2
```

## API接口

### 系统相关
- `GET /api/system/status` - 获取系统状态
- `POST /api/system/clear-cache` - 清理GPU缓存

### 图像分类
- `POST /api/classification/train` - 训练分类模型
- `POST /api/classification/predict` - 图像分类预测
- `POST /api/classification/test` - 测试分类模型
- `GET /api/classification/models` - 列出所有模型
- `GET /api/classification/download-result/{filename}` - 下载测试结果

### 图像分割
- `POST /api/segmentation/train` - 训练分割模型
- `POST /api/segmentation/predict` - 图像分割预测
- `POST /api/segmentation/test` - 测试分割模型
- `GET /api/segmentation/models` - 列出所有模型
- `GET /api/segmentation/download-result/{filename}` - 下载测试结果

## 配置说明

创建 `.env` 文件配置参数：

```env
HOST=0.0.0.0
PORT=8000
DEBUG=True

DEFAULT_EPOCHS=10
DEFAULT_BATCH_SIZE=16
DEFAULT_LR=0.001
```

## 常见问题

### 1. GPU不可用

检查ROCm安装：
```bash
/opt/rocm/bin/rocminfo
/opt/rocm/bin/rocm-smi
```

### 2. 显存不足

减少批次大小或图像尺寸：
- 调整 `DEFAULT_BATCH_SIZE` 为更小的值（如8或4）
- 调整 `MAX_IMAGE_SIZE` 为更小的值（如128或180）

### 3. 训练速度慢

确认GPU正在使用：
```bash
rocm-smi  # 查看GPU使用率
```

## 开发说明

### 添加新功能

1. 在 `app/services/` 中添加服务逻辑
2. 在 `app/routers/` 中添加路由
3. 更新前端界面和交互

### 测试

```bash
# 运行单元测试（待添加）
pytest tests/
```

## 许可证

MIT License

## 技术支持

如有问题，请提交Issue或联系开发团队。
