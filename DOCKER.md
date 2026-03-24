# Docker 部署指南

## 环境要求

- Docker 19.03+
- NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA Driver 525.60.13+
- CUDA 12.4 兼容的 GPU (如 A100)

## 安装 NVIDIA Container Toolkit

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 构建镜像

```bash
# 构建镜像
docker build -t aihoo-fast-service:latest .

# 或使用 docker-compose
docker-compose build
```

## 运行容器

### 使用 docker run

```bash
docker run -d \
  --name aihoo-tongue-diagnosis \
  --gpus all \
  --shm-size=16g \
  -p 8001:8001 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/uploads:/app/uploads \
  aihoo-fast-service:latest
```

### 使用 docker-compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 验证 GPU 状态

```bash
# 进入容器
docker exec -it aihoo-tongue-diagnosis bash

# 验证 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

## 访问服务

启动后访问: http://localhost:8001

## 常见问题

### 1. GPU 不可用

确保已安装 nvidia-container-toolkit 并重启 Docker：
```bash
sudo systemctl restart docker
```

### 2. 内存不足

增加 shared memory：
```bash
docker run --shm-size=32g ...
```

### 3. 权限问题

确保挂载目录有正确的读写权限：
```bash
chmod -R 755 data models results uploads
```
