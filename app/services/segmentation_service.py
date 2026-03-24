"""
图像分割服务
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from fastai.vision.all import *
import pandas as pd
import cv2
import numpy as np

from app.core.config import settings
from app.core.gpu_utils import check_gpu_available, initialize_best_gpu

# 配置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 镜像源
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'        # 超时时间(秒)
os.environ['CURL_CA_BUNDLE'] = ''                    # 避免SSL问题(如有)


# 模块级别的可picklable函数
_get_mask_config = {
    'train_path': None,
    'is_rgb_label': None,
    'codes': None
}


def _get_mask_fn(fn):
    """获取对应的标签文件路径 - 模块级函数，可被pickle"""
    train_path = _get_mask_config['train_path']
    is_rgb_label = _get_mask_config['is_rgb_label']

    # 尝试不同的扩展名
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        mask_path = train_path / 'labels' / f'{fn.stem}{ext}'
        if mask_path.exists():
            # 如果是RGB标签格式，需要进行转换
            if is_rgb_label:
                return _convert_rgb_to_index(mask_path)
            return mask_path
    # 如果找不到，返回.png（保持默认行为）
    return train_path / 'labels' / f'{fn.stem}.png'


def _convert_rgb_to_index(mask_path):
    """将RGB格式标签转换为索引格式，返回路径让MaskBlock处理"""
    from PIL import Image as PILImage

    # 读取图像
    img = PILImage.open(mask_path)
    img_array = np.array(img)

    # 如果是RGB，转换为灰度
    if len(img_array.shape) == 3:
        # 使用加权灰度转换
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_array

    # 将0-255映射到类别索引
    codes = _get_mask_config['codes']
    num_classes = len(codes) if codes is not None else 3
    bin_size = 256 // num_classes
    mask_indices = (gray // bin_size).astype(np.uint8)

    # 确保不超过类别数
    mask_indices = np.minimum(mask_indices, num_classes - 1)

    # 保存转换后的图像到临时位置并返回路径
    # 使用一个临时目录来存储转换后的mask
    from tempfile import mkdtemp
    import tempfile

    temp_dir = Path(_get_mask_config['train_path']) / '.temp_masks'
    temp_dir.mkdir(exist_ok=True)

    temp_mask_path = temp_dir / f"{mask_path.stem}_converted.png"
    PILImage.fromarray(mask_indices).save(temp_mask_path)

    return temp_mask_path


class SegmentationService:
    """图像分割服务类"""

    def __init__(self):
        self.model = None
        self.device = initialize_best_gpu(silent=True)  # 初始化最佳GPU（独立显卡优先，静默模式）
        self.dls = None
        self.codes = None
        self.is_training = False

    def train(
        self,
        train_path: str,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 8,
        model_name: str = "resnet34",
        resume_model: Optional[str] = None,
        pretrained: bool = False,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        训练分割模型

        Args:
            train_path: 训练数据路径
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

        self.is_training = True

        # 清理GPU缓存
        if check_gpu_available()["available"]:
            import torch
            torch.cuda.empty_cache()
            print("✓ GPU缓存已清理")

        # 初始化全局训练进度
        training_progress["segmentation"] = {
            "is_training": True,
            "status": "loading",
            "message": "正在加载数据集...",
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "valid_loss": 0.0,
            "dice": 0.0,
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

            training_progress["segmentation"].update({
                "status": status,
                "message": message,
                **safe_kwargs
            })
            if progress_callback:
                progress_callback({"status": status, "message": message, **safe_kwargs})

        try:
            update_progress("loading", "正在加载数据集...")
            
            train_path = Path(train_path)
            if not train_path.exists():
                raise FileNotFoundError(f"训练数据路径不存在: {train_path}")
            
            # 获取类别码
            codes = np.loadtxt(train_path/'codes.txt', dtype=str)
            self.codes = codes

            # 配置模块级别的函数
            _get_mask_config['train_path'] = train_path
            _get_mask_config['codes'] = codes

            # 检查图像和标签目录
            images_dir = train_path/'images'
            labels_dir = train_path/'labels'

            if not images_dir.exists():
                raise FileNotFoundError(f"图像目录不存在: {images_dir}")
            if not labels_dir.exists():
                raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

            print(f"✓ 图像目录: {images_dir}")
            print(f"✓ 标签目录: {labels_dir}")

            # 检查标签文件格式并处理RGB标签
            label_files = list(labels_dir.glob('*.png')) + list(labels_dir.glob('*.jpg')) + list(labels_dir.glob('*.jpeg'))
            is_rgb_label = False

            if label_files:
                from PIL import Image as PILImage
                sample_label = PILImage.open(label_files[0])
                label_array = np.array(sample_label)

                # 检查是否为RGB格式（3通道或最大值很大）
                if len(label_array.shape) == 3:
                    is_rgb_label = True
                    print(f"✓ 检测到RGB格式标签（3通道）")
                elif np.max(label_array) > 10:
                    # 如果最大值大于10，很可能是RGB编码（0-255）
                    is_rgb_label = True
                    print(f"✓ 检测到RGB编码的标签（最大值={np.max(label_array)}，类别数={len(codes)}）")
                else:
                    # 普通索引格式标签
                    unique_values = np.unique(label_array)
                    max_value = np.max(unique_values)
                    num_unique = len(unique_values)

                    print(f"✓ 标签文件样本: {label_files[0].name}")
                    print(f"✓ 标签中的唯一值: {unique_values}")
                    print(f"✓ 标签最大值: {max_value}")
                    print(f"✓ 标签类别数: {num_unique}")
                    print(f"✓ codes.txt类别数: {len(codes)}")

                    # 检查值范围
                    if max_value >= len(codes):
                        print(f"⚠️ 警告: 标签值 {max_value} 超出codes.txt定义的范围 [0, {len(codes)-1}]")
                        print(f"   这会导致训练失败。请检查:")
                        print(f"   1. 标签文件是否正确")
                        print(f"   2. codes.txt是否包含了所有类别")
                        print(f"   3. 类别ID是否从0开始连续编号")
                        raise ValueError(f"标签值超出范围: 检测到最大值 {max_value}，但codes.txt只定义了 {len(codes)} 个类别 [0-{len(codes)-1}]")

            # 如果标签是RGB格式，标记下来供模块级函数使用
            if is_rgb_label:
                _get_mask_config['is_rgb_label'] = True
                print(f"✓ 标签为RGB格式，将使用灰度值映射到类别索引")
                print(f"   灰度值 0-85 -> 类别 0")
                print(f"   灰度值 86-170 -> 类别 1")
                print(f"   灰度值 171-255 -> 类别 2")
                print(f"   如果这个映射不正确，请将标签转换为索引格式（0, 1, 2）")
            else:
                _get_mask_config['is_rgb_label'] = False

            # 定义数据块 - 使用更小的分辨率以适应AMD GPU
            dblock = DataBlock(
                blocks=(ImageBlock, MaskBlock(codes)),
                get_items=get_image_files,
                splitter=RandomSplitter(valid_pct=0.2, seed=42),
                get_y=_get_mask_fn,  # 使用模块级函数
                item_tfms=Resize(128, method=ResizeMethod.Squish)  # 降低分辨率到128x128
            )

            print("正在创建数据加载器...")
            dls = dblock.dataloaders(images_dir, bs=batch_size, num_workers=0)
            self.dls = dls

            print(f"✓ 数据加载器创建成功")
            print(f"✓ 训练集样本数: {len(dls.train_ds)}")
            print(f"✓ 验证集样本数: {len(dls.valid_ds)}")
            print(f"✓ 类别数量: {len(codes)}")
            print(f"✓ 类别: {list(codes)}")

            update_progress(
                "data_loaded",
                f"数据集加载完成，类别数: {len(codes)}",
                classes=list(codes),
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
                    learn = unet_learner(dls, resnet18, metrics=Dice())  # 使用更小的resnet18
                else:
                    print("使用随机初始化的模型(不从网络下载预训练权重)")
                    # 使用随机初始化的模型
                    learn = unet_learner(dls, resnet18, metrics=Dice(), pretrained=False)  # 使用更小的resnet18
            
            # 确保使用GPU
            if check_gpu_available()["available"]:
                # 打印实际使用的GPU信息
                gpu_info = check_gpu_available()
                selected_gpu = gpu_info.get("selected_device", {})
                print(f"\n✓ 图像分割训练使用GPU: {selected_gpu.get('name', 'Unknown')}")
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
                    dice = 0.0

                    # 安全地获取验证损失和Dice系数
                    if hasattr(self.learn, 'recorder') and self.learn.recorder:
                        try:
                            if self.learn.recorder.losses:
                                valid_loss = float(self.learn.recorder.losses[-1])
                            if self.learn.recorder.values and len(self.learn.recorder.values) > 0:
                                # values列表包含每个epoch的损失和指标
                                last_values = self.learn.recorder.values[-1]
                                if len(last_values) > 1:
                                    dice = float(last_values[1])
                        except (IndexError, TypeError, ValueError) as e:
                            print(f"Warning: 获取训练指标时出错: {e}")

                    # 确保所有值都是可 JSON 序列化的
                    training_progress["segmentation"].update({
                        "status": "training",
                        "message": f"正在训练: epoch {self.epoch + 1}/{self.total_epochs}",
                        "epoch": int(self.epoch + 1),
                        "train_loss": float(train_loss),
                        "valid_loss": float(valid_loss),
                        "dice": float(dice)
                    })
                    print(f"Epoch {self.epoch + 1}/{self.total_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Dice: {dice:.4f}")

            update_progress("training", "开始训练...", epochs=epochs)

            # 训练模型（使用自定义进度回调）
            learn.fit_one_cycle(epochs, lr, cbs=[TrainingProgressCallback(epochs)])
            
            # 保存模型
            update_progress("saving", "正在保存模型...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = settings.MODELS_DIR / f"segmentation_{model_name}_{timestamp}.pkl"

            # 确保模型目录存在
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"✓ 模型保存路径: {model_path}")

            learn.export(model_path)
            self.model = learn

            self.is_training = False

            final_metrics = {
                "train_loss": float(learn.recorder.final_record[0]) if hasattr(learn.recorder, 'final_record') else 0.0,
                "valid_loss": float(learn.recorder.final_record[1]) if hasattr(learn.recorder, 'final_record') else 0.0,
                "dice": float(learn.recorder.final_record[2]) if hasattr(learn.recorder, 'final_record') else 0.0
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
                "classes": list(codes),
                "metrics": {
                    "final_dice": float(learn.recorder.final_record[2])
                }
            }
            
        except Exception as e:
            self.is_training = False
            update_progress("error", f"训练失败: {str(e)}")
            raise e
    
    def predict(
        self,
        image_path: str,
        model_path: Optional[str] = None,
        return_overlay: bool = True
    ) -> Dict[str, Any]:
        """
        图像分割预测

        Args:
            image_path: 图像路径
            model_path: 模型路径
            return_overlay: 是否返回叠加图

        Returns:
            预测结果
        """
        import traceback

        try:
            print(f"[PREDICT] 开始预测，图像路径: {image_path}")
            print(f"[PREDICT] 模型路径: {model_path}")
            print(f"[PREDICT] 是否返回叠加图: {return_overlay}")

            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 加载模型
            # 处理空字符串的情况
            if model_path is not None and not model_path.strip():
                model_path = None

            if model_path:
                print(f"[PREDICT] 从文件加载模型: {model_path}")
                learn = load_learner(model_path)
            elif self.model:
                print(f"[PREDICT] 使用内存中的模型")
                learn = self.model
            else:
                # 尝试自动加载最新的模型
                models_dir = settings.MODELS_DIR
                if models_dir.exists():
                    models = list(models_dir.glob("segmentation_*.pkl"))
                    if models:
                        # 按修改时间排序，选择最新的
                        latest_model = max(models, key=lambda p: p.stat().st_mtime)
                        print(f"[PREDICT] 自动加载最新模型: {latest_model}")
                        learn = load_learner(str(latest_model))
                    else:
                        raise ValueError("没有可用的模型，请先训练或加载模型")
                else:
                    raise ValueError("没有可用的模型，请先训练或加载模型")

            print(f"[PREDICT] 模型加载成功")
            print(f"[PREDICT] learn 类型: {type(learn)}")
            print(f"[PREDICT] learn 是否有 dls: {hasattr(learn, 'dls')}")
            if hasattr(learn, 'dls'):
                print(f"[PREDICT] learn.dls 值: {learn.dls}")
                if learn.dls is not None:
                    print(f"[PREDICT] learn.dls 是否有 vocab: {hasattr(learn.dls, 'vocab')}")
                    if hasattr(learn.dls, 'vocab'):
                        print(f"[PREDICT] learn.dls.vocab: {learn.dls.vocab}")

            print(f"[PREDICT] self.dls 值: {self.dls}")
            if self.dls is not None:
                print(f"[PREDICT] self.dls 是否有 vocab: {hasattr(self.dls, 'vocab')}")
                if hasattr(self.dls, 'vocab'):
                    print(f"[PREDICT] self.dls.vocab: {self.dls.vocab}")

            # 预测
            print(f"[PREDICT] 开始加载图像...")
            img = PILImage.create(image_path)
            print(f"[PREDICT] 图像加载成功，开始预测...")
            pred_mask, pred_class, outputs = learn.predict(img)
            print(f"[PREDICT] 预测完成")
            print(f"[PREDICT] pred_mask 类型: {type(pred_mask)}")
            print(f"[PREDICT] pred_mask 形状: {pred_mask.shape}")
            print(f"[PREDICT] pred_mask 唯一值: {np.unique(np.array(pred_mask))}")

            # 获取类别信息 - 尝试多种方式获取vocab
            print(f"[PREDICT] 开始获取类别信息...")
            classes = None

            # 方法1: 尝试从learn的tfms中获取（MaskBlock的vocab）
            if hasattr(learn, 'dls') and learn.dls is not None:
                print(f"[PREDICT] 检查 learn.dls.tfms...")
                if hasattr(learn.dls, 'tfms'):
                    try:
                        # tfms是一个列表，第二个是mask的transform
                        if len(learn.dls.tfms) > 1:
                            mask_tfms = learn.dls.tfms[1]
                            print(f"[PREDICT] mask_tfms: {mask_tfms}")
                            print(f"[PREDICT] mask_tfms 类型: {type(mask_tfms)}")

                            # 尝试获取vocab
                            if hasattr(mask_tfms, 'vocab'):
                                classes = list(mask_tfms.vocab)
                                print(f"[PREDICT] 从 mask_tfms.vocab 获取类别: {classes}")
                            elif hasattr(mask_tfms, 'type'):
                                mask_block = mask_tfms.type
                                print(f"[PREDICT] mask_block: {mask_block}")
                                if hasattr(mask_block, 'vocab'):
                                    classes = list(mask_block.vocab)
                                    print(f"[PREDICT] 从 mask_block.vocab 获取类别: {classes}")
                    except Exception as e:
                        print(f"[PREDICT] 从 learn.dls.tfms 获取vocab失败: {e}")

            # 方法2: 尝试从self.dls获取
            if classes is None and self.dls is not None:
                print(f"[PREDICT] 尝试从 self.dls 获取...")
                try:
                    if hasattr(self.dls, 'tfms') and len(self.dls.tfms) > 1:
                        mask_tfms = self.dls.tfms[1]
                        if hasattr(mask_tfms, 'vocab'):
                            classes = list(mask_tfms.vocab)
                            print(f"[PREDICT] 从 self.dls.tfms[1].vocab 获取类别: {classes}")
                        elif hasattr(mask_tfms, 'type') and hasattr(mask_tfms.type, 'vocab'):
                            classes = list(mask_tfms.type.vocab)
                            print(f"[PREDICT] 从 self.dls.tfms[1].type.vocab 获取类别: {classes}")
                except Exception as e:
                    print(f"[PREDICT] 从 self.dls 获取vocab失败: {e}")

            # 方法3: 尝试从learn的y_block获取
            if classes is None and hasattr(learn, 'y_block'):
                print(f"[PREDICT] 尝试从 learn.y_block 获取...")
                try:
                    if hasattr(learn.y_block, 'vocab'):
                        classes = list(learn.y_block.vocab)
                        print(f"[PREDICT] 从 learn.y_block.vocab 获取类别: {classes}")
                except Exception as e:
                    print(f"[PREDICT] 从 learn.y_block 获取vocab失败: {e}")

            # 方法4: 从预测结果中推断
            if classes is None:
                print(f"[PREDICT] 从预测结果中推断类别")
                unique_classes = np.unique(np.array(pred_mask))
                num_classes = len(unique_classes)
                print(f"[PREDICT] 检测到 {num_classes} 个类别: {unique_classes}")
                # 尝试使用一些常见的类别名称
                common_names = ['background', 'foreground'] if num_classes == 2 else [f'class_{i}' for i in range(num_classes)]
                classes = common_names[:num_classes]
                print(f"[PREDICT] 推断的类别名称: {classes}")

            print(f"[PREDICT] 类别: {classes}")

            result = {
                "success": True,
                "image_name": Path(image_path).name,
                "classes": classes,
                "mask_shape": list(pred_mask.shape)
            }

            # 保存mask - TensorMask需要转换为PIL图像
            from PIL import Image as PILImageNative

            mask_filename = f"segmentation_mask_{timestamp}.png"
            mask_path = settings.RESULTS_DIR / mask_filename

            # 将TensorMask转换为numpy数组再转为PIL图像
            mask_array = np.array(pred_mask)
            # 确保是uint8类型
            if mask_array.dtype != np.uint8:
                mask_array = mask_array.astype(np.uint8)
            pil_mask = PILImageNative.fromarray(mask_array, mode='L')
            pil_mask.save(mask_path)

            result["mask_path"] = f"/results/{mask_filename}"
            print(f"[PREDICT] Mask已保存到: {mask_path}")

            # 生成叠加图和各类别独立图像
            if return_overlay:
                print(f"[PREDICT] 开始生成叠加图和类别图像...")
                overlay_filename, class_images = self._create_overlay(image_path, pred_mask, timestamp)
                result["overlay_path"] = f"/results/{overlay_filename}"
                result["class_images"] = class_images
                print(f"[PREDICT] 叠加图文件名: {overlay_filename}")
                print(f"[PREDICT] 生成了 {len(class_images)} 个类别图像")

            # 计算各类别像素占比
            print(f"[PREDICT] 计算类别分布...")
            mask_array = np.array(pred_mask)
            class_distribution = []
            for i, cls in enumerate(classes):
                pixel_count = np.sum(mask_array == i)
                percentage = (pixel_count / mask_array.size) * 100
                class_distribution.append({
                    "class": cls,
                    "pixel_count": int(pixel_count),
                    "percentage": round(percentage, 2)
                })
            result["class_distribution"] = class_distribution

            print(f"[PREDICT] 预测成功完成")
            return result

        except Exception as e:
            print(f"[PREDICT] 发生错误: {e}")
            print(f"[PREDICT] 错误类型: {type(e).__name__}")
            print(f"[PREDICT] 详细堆栈:")
            traceback.print_exc()
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
        测试分割模型
        
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
            
            test_path = Path(test_path)
            images_path = test_path / 'images'
            labels_path = test_path / 'labels'
            
            if not images_path.exists():
                raise FileNotFoundError(f"测试图像路径不存在: {images_path}")
            
            image_files = get_image_files(images_path)
            total = len(image_files)
            
            if progress_callback:
                progress_callback({
                    "status": "testing",
                    "message": f"开始测试，共 {total} 张图片",
                    "total": total,
                    "processed": 0
                })
            
            results = []
            total_dice = 0
            total_iou = 0
            
            for i, img_path in enumerate(image_files):
                # 获取真实mask
                mask_path = labels_path / f'{img_path.stem}.png'
                if mask_path.exists():
                    true_mask = PILMask.create(mask_path)
                else:
                    continue
                
                # 预测
                pred_mask, _, _ = learn.predict(img_path)
                
                # 计算指标
                dice = self._calculate_dice(pred_mask, true_mask)
                iou = self._calculate_iou(pred_mask, true_mask)
                
                total_dice += dice
                total_iou += iou
                
                results.append({
                    "image_name": img_path.name,
                    "dice": dice,
                    "iou": iou
                })
                
                # 进度回调
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback({
                        "status": "testing",
                        "message": f"已处理 {i + 1}/{total} 张图片",
                        "total": total,
                        "processed": i + 1
                    })
            
            # 计算平均指标
            avg_dice = total_dice / len(results) if results else 0
            avg_iou = total_iou / len(results) if results else 0
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = settings.RESULTS_DIR / f"segmentation_test_{timestamp}.csv"
            df = pd.DataFrame(results)
            df.to_csv(result_path, index=False, encoding='utf-8-sig')
            
            if progress_callback:
                progress_callback({
                    "status": "completed",
                    "message": "测试完成！",
                    "avg_dice": avg_dice,
                    "avg_iou": avg_iou,
                    "result_path": str(result_path)
                })
            
            return {
                "success": True,
                "avg_dice": avg_dice,
                "avg_iou": avg_iou,
                "total": len(results),
                "result_path": str(result_path),
                "results": results[:10]
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
    
    def _create_overlay(self, image_path: str, mask, timestamp: str) -> str:
        """创建mask叠加图，返回文件名"""
        import matplotlib.pyplot as plt

        # 读取原始图像
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整mask大小
        mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # 创建叠加图
        overlay = img.copy()
        alpha = 0.5

        # 为每个类别应用不同颜色
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
        ]

        for i in range(len(colors)):
            overlay[mask_resized == i] = overlay[mask_resized == i] * (1 - alpha) + np.array(colors[i]) * alpha

        # 保存叠加图
        overlay_filename = f"segmentation_overlay_{timestamp}.png"
        overlay_path = settings.RESULTS_DIR / overlay_filename
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # 保存每个类别的独立图像
        from PIL import Image as PILImageNative
        class_images = []

        # 获取实际存在的类别
        unique_classes = np.unique(mask_resized)
        print(f"[CREATE_OVERLAY] 实际类别数: {len(unique_classes)}, 类别: {unique_classes}")

        for class_id in unique_classes:
            # 创建该类别的二值mask
            class_mask = np.zeros_like(mask_resized, dtype=np.uint8)
            class_mask[mask_resized == class_id] = 255

            # 将mask应用到原始图像，提取该类别的区域
            class_image = img.copy().astype(np.float32)
            mask_float = class_mask.astype(np.float32) / 255.0
            mask_float = np.stack([mask_float] * 3, axis=-1)

            # 只保留该类别的像素，其他区域设为黑色
            class_image = class_image * mask_float
            class_image = np.clip(class_image, 0, 255).astype(np.uint8)

            # 保存类别图像
            class_filename = f"segmentation_class_{class_id}_{timestamp}.png"
            class_path = settings.RESULTS_DIR / class_filename
            cv2.imwrite(str(class_path), cv2.cvtColor(class_image, cv2.COLOR_RGB2BGR))
            class_images.append({
                "class_id": int(class_id),
                "class_name": f"class_{class_id}",
                "image_path": f"/results/{class_filename}"
            })

        return overlay_filename, class_images
    
    def _calculate_dice(self, pred_mask, true_mask):
        """计算Dice系数"""
        pred_array = np.array(pred_mask)
        true_array = np.array(true_mask)
        
        intersection = np.sum(pred_array == true_array)
        dice = 2 * intersection / (pred_array.size + true_array.size)
        
        return dice
    
    def _calculate_iou(self, pred_mask, true_mask):
        """计算IoU"""
        pred_array = np.array(pred_mask)
        true_array = np.array(true_mask)
        
        intersection = np.sum(pred_array == true_mask)
        union = pred_array.size + true_array.size - intersection
        iou = intersection / union if union > 0 else 0
        
        return iou


# 全局训练进度存储
training_progress = {
    "segmentation": {
        "is_training": False,
        "status": "idle",
        "message": "",
        "epoch": 0,
        "total_epochs": 0,
        "train_loss": 0.0,
        "valid_loss": 0.0,
        "dice": 0.0,
        "start_time": "",
        "classes": [],
        "train_count": 0,
        "valid_count": 0
    }
}

# 全局服务实例
segmentation_service = SegmentationService()
