import os
from ultralytics import YOLO

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Model Configuration
    # You can use standard Ultralytics weights like 'yolov8n-seg.pt' which will auto-download
    # or specify a local path to pretrained weights.
    model_name = "yolov8n-seg.pt" 
    # model_name = "/path/to/custom/weights.pt"
    
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # 2. Start Training
    results = model.train(
        # Dataset Config Path
        data=os.path.join(project_root, "dataset", "data_aug.yaml"),

        # Explicitly specify task as instance segmentation
        task="segment",
        
        # Input image size (resized to this before training)
        imgsz=640,
        
        # Total Epochs
        epochs=100,            
        
        # Batch Size. Adjust according to your GPU memory.
        batch=16,
        
        # GPU Device Index. '0' for first GPU. Use 'cpu' for CPU.
        device='0',
        
        # Project folder for saving results
        project="runs_seg_train",
        
        # Experiment name (results saved in runs_seg_train/exp_name)
        name="experiment_v1",
        
        # Load pretrained weights
        pretrained=True,
        
        # Early Stopping patience
        patience=20,
        
        # Auto-save model checkpoints
        save=True,
        
        # --- Data Augmentation Config (Optimized for Offline Augmented Data) ---
        mosaic=1.0,     # Enable Mosaic augmentation (helps with small objects)
        
        # Geometric Transforms: 
        # Since we applied offline augmentation (simulating distant view), 
        # we reduce online geometric distortions to avoid "double distortion".
        translate=0.05, # Slight translation
        scale=0.1,      # Slight scaling (default 0.5 is too high for pre-scaled data)
        fliplr=0.5,     # Enable horizontal flip
        degrees=0.0,    # Disable rotation (handled offline)
        
        # Color Transforms: Simulate lighting conditions
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

if __name__ == "__main__":
    main()
