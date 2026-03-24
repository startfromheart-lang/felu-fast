"""
数据集准备示例脚本
用于创建示例数据集结构
"""
import os
import shutil
from pathlib import Path


def prepare_classification_dataset():
    """创建图像分类数据集示例结构"""
    base_path = Path("data/classification_example")
    train_path = base_path / "train"
    valid_path = base_path / "valid"
    
    # 创建目录
    for cat in ["cat", "dog"]:
        (train_path / cat).mkdir(parents=True, exist_ok=True)
        (valid_path / cat).mkdir(parents=True, exist_ok=True)
    
    print("图像分类数据集结构已创建:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {valid_path}")
    print("\n请按以下结构组织您的数据:")
    print(f"  {train_path}/")
    print("    ├── cat/")
    print("    │   ├── cat1.jpg")
    print("    │   └── cat2.jpg")
    print("    └── dog/")
    print("        ├── dog1.jpg")
    print("        └── dog2.jpg")
    print(f"  {valid_path}/")
    print("    ├── cat/")
    print("    └── dog/")


def prepare_segmentation_dataset():
    """创建图像分割数据集示例结构"""
    base_path = Path("data/segmentation_example")
    images_path = base_path / "images"
    labels_path = base_path / "labels"
    
    # 创建目录
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # 创建codes.txt
    codes_path = base_path / "codes.txt"
    codes_path.write_text("background\nperson\ncar\n")
    
    print("\n图像分割数据集结构已创建:")
    print(f"  数据集: {base_path}")
    print(f"  图像目录: {images_path}")
    print(f"  标签目录: {labels_path}")
    print(f"  类别文件: {codes_path}")
    print("\n请按以下结构组织您的数据:")
    print(f"  {base_path}/")
    print("    ├── images/")
    print("    │   ├── img1.jpg")
    print("    │   └── img2.jpg")
    print("    ├── labels/")
    print("    │   ├── img1.png")
    print("    │   └── img2.png")
    print("    └── codes.txt")


def main():
    print("=" * 50)
    print("数据集准备工具")
    print("=" * 50)
    
    # 创建数据目录
    Path("data").mkdir(exist_ok=True)
    
    print("\n选择要准备的数据集类型:")
    print("1. 图像分类数据集")
    print("2. 图像分割数据集")
    print("3. 全部创建")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == "1":
        prepare_classification_dataset()
    elif choice == "2":
        prepare_segmentation_dataset()
    elif choice == "3":
        prepare_classification_dataset()
        prepare_segmentation_dataset()
    else:
        print("无效选项")
        return
    
    print("\n" + "=" * 50)
    print("数据集结构创建完成！")
    print("=" * 50)
    print("\n下一步:")
    print("1. 将您的数据集文件放入对应目录")
    print("2. 启动服务: bash start.sh")
    print("3. 访问 http://localhost:8000")


if __name__ == "__main__":
    main()
