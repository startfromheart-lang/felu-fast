import shutil
import os
import sys

# 获取site-packages目录
import site
site_packages = site.getsitepackages()
print("Site packages directories:")
for sp in site_packages:
    print(f"  {sp}")

# 要清理的包目录
torch_packages = [
    'torch',
    'torch-2.10.0.dist-info',
    'torch-2.11.0.dist-info',
    'torch-2.11.0+cu128.dist-info',
    'torchaudio',
    'torchaudio-2.11.0.dist-info',
    'torchaudio-2.11.0+cu128.dist-info',
    'torchvision',
    'torchvision-0.25.0.dist-info',
    'torchvision-0.26.0.dist-info',
    'torchvision-0.26.0+cu128.dist-info',
    'torchgen',
    'functorch',
]

# 在所有site-packages目录中清理
cleaned = []
for sp in site_packages:
    for pkg in torch_packages:
        pkg_path = os.path.join(sp, pkg)
        if os.path.exists(pkg_path):
            try:
                shutil.rmtree(pkg_path)
                cleaned.append(pkg_path)
                print(f"Removed: {pkg_path}")
            except Exception as e:
                print(f"Failed to remove {pkg_path}: {e}")

print(f"\nCleaned {len(cleaned)} directories")
print("Done!")
