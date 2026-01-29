from ultralytics import YOLO
import os
import cv2
import glob
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # --- Configuration ---
    # Path to trained weights
    MODEL_PATH = project_root / "runs_seg_train" / "experiment_v1" / "weights" / "best.pt"
    # Or use standard weight:
    # MODEL_PATH = "yolov8n-seg.pt"
    
    INPUT_DIR = str(project_root / "data")
    OUTPUT_DIR = str(project_root / "inference_results")
    
    # Supported Formats
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # --- Setup ---
    if not os.path.exists(MODEL_PATH) and not str(MODEL_PATH).endswith(".pt"):
         # Simple check, though YOLO auto-downloads if simple name
         print(f"Warning: Weights not found at ({MODEL_PATH})")
         print("Falling back to yolov8n-seg.pt ...")
         MODEL_PATH = "yolov8n-seg.pt"

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))
    print(f"Found {len(files)} files in {INPUT_DIR}")
    
    for filepath in files:
        path_obj = Path(filepath)
        ext = path_obj.suffix.lower()
        
        if ext in IMG_EXTS:
            process_image(model, filepath, OUTPUT_DIR)
        elif ext in VID_EXTS:
            process_video(model, filepath, OUTPUT_DIR)
        else:
            print(f"Skipping unsupported file: {filepath}")

    print(f"\nInference Complete! Results saved to: {os.path.abspath(OUTPUT_DIR)}")

def process_image(model, img_path, out_dir):
    print(f"Processing Image: {img_path} ...", end=" ")
    filename = os.path.basename(img_path)
    save_path = os.path.join(out_dir, filename)
    
    # Run inference
    results = model.predict(img_path, conf=0.25, iou=0.45)
    
    for r in results:
        # Plot results on the image
        im_array = r.plot()  # BGR numpy array
        cv2.imwrite(save_path, im_array)
        
    print(f"-> Saved to {save_path}")

def process_video(model, vid_path, out_dir):
    print(f"Processing Video: {vid_path} ...")
    filename = os.path.basename(vid_path)
    save_path = os.path.join(out_dir, "out_" + filename)
    
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error opening video {vid_path}")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference on frame
        # stream=True for long videos to manage memory
        results = model.predict(frame, conf=0.25, verbose=False)
        
        for r in results:
            annotated_frame = r.plot()
            out.write(annotated_frame)
            
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"  Frame {frame_count}/{total_frames}", end='\r')
            
    cap.release()
    out.release()
    print(f"\n-> Saved video to {save_path}")

if __name__ == "__main__":
    main()
