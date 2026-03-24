"""
图像分类服务
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from fastai.vision.all import *
import pandas as pd
import torch

from app.core.config import settings
from app.core.gpu_utils import check_gpu_available, initialize_best_gpu

# 配置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 镜像源
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'        # 超时时间(秒)
os.environ['CURL_CA_BUNDLE'] = ''                    # 避免SSL问题(如有)



class ClassificationService:
    """图像分类服务类"""
    
    def __init__(self):
        self.model = None
        self.device = initialize_best_gpu(silent=True)  # 初始化最佳GPU（独立显卡优先，静默模式）
        self.dls = None
        self.classes = []
        self.is_training = False
    
    def train(
        self,
        train_path: str,
        valid_path: Optional[str] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 16,
        model_name: str = "resnet18",
        resume_model: Optional[str] = None,
        pretrained: bool = False,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        训练分类模型

        Args:
            train_path: 训练数据路径
            valid_path: 验证数据路径（可选）
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小
            model_name: 模型名称
            resume_model: 恢复训练的模型路径
            progress_callback: 进度回调函数

        Returns:
            训练结果字典
        """
        from datetime import datetime

        print("=" * 60)
        print("[train] 开始训练（文件夹模式）")
        print(f"[train] 训练路径: {train_path}")
        print(f"[train] 验证路径: {valid_path}")
        print(f"[train] 训练轮数: {epochs}")
        print(f"[train] 学习率: {lr}")
        print(f"[train] 批次大小: {batch_size}")
        print(f"[train] 模型名称: {model_name}")
        print(f"[train] 预训练: {pretrained}")
        print("=" * 60)

        self.is_training = True

        # 初始化全局训练进度
        training_progress["classification"] = {
            "is_training": True,
            "status": "loading",
            "message": "正在加载数据集...",
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "valid_loss": 0.0,
            "accuracy": 0.0,
            "start_time": datetime.now().isoformat()
        }
        
        def update_progress(status, message, **kwargs):
            """更新全局训练进度"""
            # 确保所有值都是可 JSON 序列化的
            safe_kwargs = {}
            for key, value in kwargs.items():
                # 转换 numpy 类型为 Python 原生类型
                if value is None:
                    safe_kwargs[key] = None
                elif hasattr(value, 'tolist'):
                    safe_kwargs[key] = value.tolist()
                elif hasattr(value, 'item'):
                    safe_kwargs[key] = float(value.item())
                elif isinstance(value, (np.integer, np.floating)):
                    safe_kwargs[key] = float(value)
                elif isinstance(value, np.ndarray):
                    safe_kwargs[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    safe_kwargs[key] = list(value)
                elif isinstance(value, dict):
                    safe_kwargs[key] = value
                elif isinstance(value, (str, int, float, bool)):
                    safe_kwargs[key] = value
                else:
                    # 对于未知类型，尝试转换为字符串
                    try:
                        safe_kwargs[key] = str(value)
                    except Exception as e:
                        safe_kwargs[key] = f"<unserializable: {type(value).__name__}>"

            training_progress["classification"].update({
                "status": status,
                "message": message,
                **safe_kwargs
            })
            if progress_callback:
                progress_callback({"status": status, "message": message, **safe_kwargs})
        
        try:
            update_progress("loading", "正在加载数据集...")

            # 加载数据
            train_path = Path(train_path)
            print(f"✓ 训练路径: {train_path}")
            print(f"✓ 路径存在: {train_path.exists()}")

            if not train_path.exists():
                raise FileNotFoundError(f"训练数据路径不存在: {train_path}")

            print(f"✓ 验证路径: {valid_path}")
            if valid_path:
                valid_path = Path(valid_path)
                print(f"✓ 验证路径存在: {valid_path.exists()}")

            # 使用 ImageDataLoaders.from_folder 创建数据加载器（更稳定的方式）
            print("正在创建数据加载器...")
            try:
                # 尝试最简单的配置
                if valid_path and Path(valid_path).exists():
                    # 如果有独立的验证集路径
                    print("使用独立的训练/验证集模式")
                    dls = ImageDataLoaders.from_folder(
                        train_path.parent,
                        train='train',
                        valid='valid',
                        seed=42,
                        item_tfms=Resize(224, method=ResizeMethod.Squish),
                        bs=batch_size,
                        num_workers=0
                    )
                else:
                    # 使用随机分割
                    print("使用随机分割模式")
                    dls = ImageDataLoaders.from_folder(
                        train_path,
                        valid_pct=0.2,
                        seed=42,
                        item_tfms=Resize(224, method=ResizeMethod.Squish),
                        bs=batch_size,
                        num_workers=0
                    )
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"创建数据加载器失败详细错误:\n{error_detail}")
                raise Exception(f"创建数据加载器失败: {str(e)}")

            self.dls = dls
            self.classes = dls.vocab

            print(f"✓ 数据加载器创建成功")
            print(f"✓ 训练集样本数: {len(dls.train_ds)}")
            print(f"✓ 验证集样本数: {len(dls.valid_ds)}")
            print(f"✓ 类别数量: {len(dls.vocab)}")
            print(f"✓ 类别: {list(dls.vocab)}")

            update_progress(
                "data_loaded",
                f"数据集加载完成，类别数: {len(self.classes)}",
                classes=list(self.classes),
                train_count=len(dls.train_ds),
                valid_count=len(dls.valid_ds)
            )
            
            # 创建或加载模型
            if resume_model and Path(resume_model).exists():
                update_progress("loading_model", "正在加载已有模型...")
                learn = load_learner(resume_model)
                learn.dls = dls
            else:
                update_progress("creating_model", "正在创建模型...")

                # 如果不使用预训练权重,则从零开始训练
                if pretrained:
                    update_progress("downloading_pretrained", "正在下载预训练模型...")
                    print(f"\n✓ 使用Hugging Face镜像源: {os.environ.get('HF_ENDPOINT', '官方源')}")
                    print("✓ 下载超时时间:", os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '60'), "秒")
                    print("✓ 正在下载预训练模型,如果网络较慢可能需要等待...")
                    learn = vision_learner(dls, model_name, metrics=accuracy)
                else:
                    print("使用随机初始化的模型(不从网络下载预训练权重)")
                    # 使用随机初始化的模型
                    learn = vision_learner(dls, model_name, metrics=accuracy, pretrained=False)
            
            # 确保使用GPU
            if check_gpu_available()["available"]:
                # 打印实际使用的GPU信息
                gpu_info = check_gpu_available()
                selected_gpu = gpu_info.get("selected_device", {})
                print(f"\n✓ 图像分类训练使用GPU: {selected_gpu.get('name', 'Unknown')}")
                print(f"✓ 显存: {selected_gpu.get('memory_gb', 'Unknown')} GB")
                print(f"✓ 设备ID: {self.device}")
                print()

                learn.model = learn.model.to(self.device)
                learn.dls.to(self.device)
            
            # 创建自定义训练回调
            class TrainingProgressCallback(Callback):
                def __init__(self, total_epochs):
                    super().__init__()
                    self.total_epochs = total_epochs

                def after_epoch(self):
                    # 更新全局训练进度
                    train_loss = float(self.loss.item()) if hasattr(self.loss, 'item') else float(self.loss)
                    valid_loss = 0.0
                    accuracy = 0.0

                    # 安全地获取验证损失和准确率
                    if hasattr(self.learn, 'recorder') and self.learn.recorder:
                        try:
                            if self.learn.recorder.losses:
                                valid_loss = float(self.learn.recorder.losses[-1])
                            if self.learn.recorder.values and len(self.learn.recorder.values) > 0:
                                # values列表包含每个epoch的损失和指标
                                last_values = self.learn.recorder.values[-1]
                                if len(last_values) > 1:
                                    accuracy = float(last_values[1])
                        except (IndexError, TypeError, ValueError) as e:
                            print(f"Warning: 获取训练指标时出错: {e}")

                    # 确保所有值都是可 JSON 序列化的
                    training_progress["classification"].update({
                        "status": "training",
                        "message": f"正在训练: epoch {self.epoch + 1}/{self.total_epochs}",
                        "epoch": int(self.epoch + 1),
                        "train_loss": float(train_loss),
                        "valid_loss": float(valid_loss),
                        "accuracy": float(accuracy)
                    })
                    print(f"Epoch {self.epoch + 1}/{self.total_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")

            update_progress("training", "开始训练...", epochs=epochs)

            # 训练模型（使用自定义进度回调）
            learn.fit_one_cycle(epochs, lr, cbs=[TrainingProgressCallback(epochs)])
            
            # 保存模型
            update_progress("saving", "正在保存模型...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = settings.MODELS_DIR / f"classification_{model_name}_{timestamp}.pkl"

            # 确保模型目录存在
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"✓ 模型保存路径: {model_path}")

            learn.export(model_path)
            self.model = learn

            self.is_training = False

            final_metrics = {
                "train_loss": float(learn.recorder.final_record[0]) if hasattr(learn.recorder, 'final_record') else 0.0,
                "valid_loss": float(learn.recorder.final_record[1]) if hasattr(learn.recorder, 'final_record') else 0.0,
                "accuracy": float(learn.recorder.final_record[2]) if hasattr(learn.recorder, 'final_record') else 0.0
            }

            update_progress(
                "completed",
                "训练完成！",
                model_path=str(model_path),
                final_metrics=final_metrics
            )
            
            return {
                "success": True,
                "model_path": str(model_path),
                "classes": list(self.classes),
                "metrics": {
                    "final_accuracy": float(learn.recorder.final_record[2])
                }
            }
            
        except Exception as e:
            self.is_training = False
            update_progress("error", f"训练失败: {str(e)}")
            raise e
    
    def predict(
        self,
        image_path: str,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        图像分类预测

        Args:
            image_path: 图像路径
            model_path: 模型路径

        Returns:
            预测结果
        """
        try:
            # 处理空字符串的情况
            if model_path is not None and not model_path.strip():
                model_path = None

            # 加载模型
            if model_path:
                learn = load_learner(model_path)
            elif self.model:
                learn = self.model
            else:
                # 尝试自动加载最新的模型
                from pathlib import Path
                models_dir = settings.MODELS_DIR
                if models_dir.exists():
                    models = list(models_dir.glob("classification_*.pkl"))
                    if models:
                        # 按修改时间排序，选择最新的
                        latest_model = max(models, key=lambda p: p.stat().st_mtime)
                        print(f"自动加载最新模型: {latest_model}")
                        learn = load_learner(str(latest_model))
                    else:
                        raise ValueError("没有可用的模型，请先训练或加载模型")
                else:
                    raise ValueError("没有可用的模型，请先训练或加载模型")
            
            # 预测
            img = PILImage.create(image_path)
            pred_class, pred_idx, outputs = learn.predict(img)
            
            # 获取所有类别的概率
            probs = torch.softmax(outputs, dim=0)
            class_probs = []
            for i, cls in enumerate(learn.dls.vocab):
                class_probs.append({
                    "class": cls,
                    "probability": float(probs[i])
                })
            
            return {
                "success": True,
                "predicted_class": pred_class,
                "confidence": float(probs[pred_idx]),
                "all_probabilities": sorted(class_probs, key=lambda x: x["probability"], reverse=True)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test(
        self,
        test_path: str,
        model_path: Optional[str] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        测试模型

        Args:
            test_path: 测试数据路径
            model_path: 模型路径
            progress_callback: 进度回调

        Returns:
            测试结果
        """
        try:
            # 加载模型
            if model_path:
                learn = load_learner(model_path)
            elif self.model:
                learn = self.model
            else:
                raise ValueError("没有可用的模型，请先训练或加载模型")

            # 获取测试图像
            test_path = Path(test_path)
            if not test_path.exists():
                raise FileNotFoundError(f"测试数据路径不存在: {test_path}")

            image_files = get_image_files(test_path)
            total = len(image_files)

            if progress_callback:
                progress_callback({
                    "status": "testing",
                    "message": f"开始测试，共 {total} 张图片",
                    "total": total,
                    "processed": 0
                })

            results = []
            correct = 0

            for i, img_path in enumerate(image_files):
                # 预测
                pred_class, pred_idx, outputs = learn.predict(img_path)
                probs = torch.softmax(outputs, dim=0)

                # 真实类别（从父目录名获取）
                true_class = img_path.parent.name

                # 判断是否正确
                is_correct = (pred_class == true_class)
                if is_correct:
                    correct += 1

                results.append({
                    "image_name": img_path.name,
                    "predicted_class": pred_class,
                    "true_class": true_class,
                    "confidence": float(probs[pred_idx]),
                    "correct": is_correct
                })

                # 进度回调
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback({
                        "status": "testing",
                        "message": f"已处理 {i + 1}/{total} 张图片",
                        "total": total,
                        "processed": i + 1
                    })

            # 计算准确率
            accuracy = correct / total if total > 0 else 0

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = settings.RESULTS_DIR / f"classification_test_{timestamp}.csv"
            df = pd.DataFrame(results)
            df.to_csv(result_path, index=False, encoding='utf-8-sig')

            if progress_callback:
                progress_callback({
                    "status": "completed",
                    "message": "测试完成！",
                    "accuracy": accuracy,
                    "result_path": str(result_path)
                })

            return {
                "success": True,
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "result_path": str(result_path),
                "results": results[:10]  # 返回前10条结果示例
            }

        except Exception as e:
            if progress_callback:
                progress_callback({
                    "status": "error",
                    "message": f"测试失败: {str(e)}"
                })
            return {
                "success": False,
                "error": str(e)
            }


# 全局训练进度存储
training_progress = {
    "classification": {
        "is_training": False,
        "status": "idle",
        "message": "",
        "epoch": 0,
        "total_epochs": 0,
        "train_loss": 0.0,
        "valid_loss": 0.0,
        "accuracy": 0.0,
        "start_time": "",
        "classes": [],
        "train_count": 0,
        "valid_count": 0
    }
}

# 全局服务实例
classification_service = ClassificationService()
