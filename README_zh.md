# SAYOLO-Seg: SAM 辅助的 YOLO 实例分割标注工具

[中文文档](README_zh.md) | [English](README.md)

SAYOLO-Seg 是一个专为 **实例分割 (Instance Segmentation)** 任务设计的高效、半自动化标注工具。它利用 **Segment Anything Model (SAM)** 的强大能力，通过简单的点提示即可生成高质量的多边形掩码，显著加速 YOLOv8/v11 分割模型的数据集制作过程。

## ✨ 特性

*   **⚡ SAM 驱动标注**: 点击即可生成掩码，无需繁琐的手动画多边形。
*   **🎯 YOLO-Seg 兼容**: 自动导出为标准的 YOLO 分割格式 (`class x1 y1 ...` 归一化坐标)。
*   **🧩 智能碎片处理**: 智能算法处理物体遮挡（例如碗将桌子遮挡成两部分的情况），同时过滤掉噪点。
*   **🛠️ 完整流水线**: 包含从 **数据增强**（模拟机器人视角）、**数据集划分**、**模型训练** 到 **推理测试** 的全套脚本。
*   **🔒 隐私保护**: UI 界面隐藏了服务器绝对路径，适合演示和分享。

## 📺 演示

![Demo](demo.gif)

*观看视频了解如何在几秒钟内标注复杂物体。*

## 📂 项目结构

```text
SAYOLO-Seg/
├── app.py                  # Gradio 主程序
├── classes.txt             # 类别定义文件
├── requirements.txt        # 依赖包列表
├── data/                   # 原始图片存放处
├── output/                 # 标注结果保存处
│   ├── images/
│   └── labels/             # YOLO 格式的 .txt 文件
├── weights/                # 存放 SAM 模型权重 (可选)
└── scripts/
    ├── split_dataset.py    # 划分训练集/验证集
    ├── augment_data.py     # 离线数据增强 (Albumentations)
    ├── train.py            # 训练 YOLOv8-Seg 模型
    └── inference.py        # 在新图片上运行推理
```

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/UniqueYwY/SAYOLO-Seg.git
cd SAYOLO-Seg
pip install -r requirements.txt
```

### 2. 下载权重

下载 Segment Anything Model (SAM) 的权重文件 (例如 `sam_vit_b_01ec64.pth` 或 MobileSAM) 并放入 `weights/` 目录。

> 如果需要，可以在 `app.py` 中配置模型路径。

### 3. 配置类别

修改项目根目录下的 `classes.txt` 文件。添加你的物体名称，每行一个。

```text
person
car
background
```

### 4. 运行标注工具

```bash
python app.py
```

*   在浏览器打开 `http://localhost:7860`
*   **Step 1**: 在列表中选择一张图片。
*   **Step 2**: 选择一个类别。
*   **Step 3**: 点击物体添加正向点 (+)。右键点击或勾选 "Remove" 以排除区域 (-)。
*   **Step 4**: 点击 **"Add Mask"** 确认该物体。
*   **Step 5**: 完成后点击 **"Save Image"**。

## 🛠️ 工作流：从标注到训练

1.  **标注**: 标注图片。结果保存在 `output/`。
    ```bash
    python app.py
    ```
2.  **划分**: 将 `output/` 整理为 `dataset/` 下标准的 `train/val` 结构。
    ```bash
    python scripts/split_dataset.py
    ```
3.  **增强 (可选)**: 生成增强样本（例如模拟机器人远视视角）。
    ```bash
    python scripts/augment_data.py
    ```
4.  **训练**: 在你的数据集上微调 YOLOv8-Seg 模型。
    ```bash
    python scripts/train.py
    ```

## 📝 许可证

本项目基于 MIT 许可证开源。
