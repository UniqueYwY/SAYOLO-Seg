import os

# --- CONFIGURE THESE BEFORE RUNNING ---

# 1. Project Root
# (Automatically detects the folder where this script is located)
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Model Configuration
# Download SAM weights (e.g., sam_b.pt, sam_l.pt, sam_h.pt, or MobileSAM) 
# and place them in a 'weights' folder or update this path.
# Default assumes a 'weights' directory in the project root.
MODEL_PATH = os.path.join(WORKSPACE_DIR, "weights", "sam_b.pt") 
# Config path is optional for some SAM versions, but required for others (like SAM2/3)
# If using standard SAM/MobileSAM, this might not be needed.
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "weights", "mobile_sam.json")

# 3. Input/Output Configuration
INPUT_IMAGES_DIR = os.path.join(WORKSPACE_DIR, "data") 
OUTPUT_DATA_DIR = os.path.join(WORKSPACE_DIR, "output")

# --------------------------------------

# Derived Paths (Do not need to change usually)
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DATA_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DATA_DIR, "labels")
DATA_YAML_PATH = os.path.join(OUTPUT_DATA_DIR, "data.yaml")
NOTES_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "notes.json")
CLASSES_FILE = os.path.join(WORKSPACE_DIR, "classes.txt")

# 5. Temporary Directory
# Use system temp dir so we don't pollute the project folder
import tempfile
_GRADIO_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gradio_cache_sayolo")

# --- INITIALIZATION ---

# Initialize environment
os.makedirs(_GRADIO_TEMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _GRADIO_TEMP_DIR

# Ensure necessary directories exist
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Create default class file if not exists
if not os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "w") as f:
        f.write("object\nbackground")

import gradio as gr
import cv2
import numpy as np
import torch
import glob
import shutil
from pathlib import Path
import json
import time

# --- Model Wrapper ---
class SAMWrapper:
    def __init__(self, model_path, config_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.predictor = None
        self.model_path = model_path
        self.config_path = config_path
        self.is_ready = False
        
        print(f"Initializing SAM Wrapper on {device}...")
        self._try_load_model()

    def _try_load_model(self):
        # Strategy 1: transformers
        try:
            print("Attempting to load via transformers...")
            from transformers import AutoModel
            model_dir = os.path.dirname(self.model_path)
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
            self.is_ready = True
            print("Loaded via transformers!")
            return
        except Exception as e:
            print(f"Transformers load failed: {e}")

        # Strategy 2: ultralytics
        try:
            print("Attempting to load via ultralytics...")
            from ultralytics import SAM
            self.model = SAM(self.model_path)
            self.is_ready = True
            print("Loaded via ultralytics!")
            return
        except Exception as e:
            print(f"Ultralytics load failed: {e}")

        print("WARNING: Could not load SAM model automatically with known libraries.")
        print("Running in UI-ONLY mode (Mock Model).")
        self.is_ready = False

    def set_image(self, image_np):
        self.current_image = image_np

    def predict(self, points, labels):
        if not self.is_ready:
            if not hasattr(self, 'current_image') or self.current_image is None:
                return None
            h, w = self.current_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            if len(points) > 0:
                cv2.circle(mask, tuple(points[-1]), 50, 1, -1)
            return mask.astype(bool)

        try:
            if hasattr(self.model, 'predict'):
                results = self.model(self.current_image, points=[points], labels=[labels], retina_masks=True, stream=False)
                if results and len(results) > 0:
                    res = results[0]
                    if res.masks is not None:
                        return res.masks.data[0].cpu().numpy().astype(bool)
                return np.zeros(self.current_image.shape[:2], dtype=bool)

        except Exception as e:
            print(f"Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            h, w = self.current_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            if len(points) > 0:
                cv2.circle(mask, tuple(points[-1]), 20, 1, -1)
            return mask.astype(bool)
        
        return np.zeros(self.current_image.shape[:2], dtype=bool)


# --- Helper Functions ---
def load_classes():
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return lines
    return ["object"]

def get_image_list():
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    # Use configured INPUT_IMAGES_DIR
    for ext in exts:
        files.extend(glob.glob(os.path.join(INPUT_IMAGES_DIR, ext)))
        files.extend(glob.glob(os.path.join(INPUT_IMAGES_DIR, ext.upper())))
    return sorted(files)

def mask_to_polygon(mask):
    # Use RETR_EXTERNAL to retrieve only the outermost contours (ignoring internal holes, consistent with standard YOLO format)
    # If "doughnut" shapes (objects with holes) need to be handled, geometric algorithms are typically required to cut the hole 
    # and form a single connected domain. We align with YOLO custom to keep it simple.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    H, W = mask.shape
    
    if not contours:
        return polygons

    # Strategy Improvement: Keep all "significant" regions, filter only tiny noise.
    # Previously "Keep Largest Only" caused issues with objects split by occlusion.
    
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    
    for contour, area in zip(contours, areas):
        # 1. Absolute Area Filter: Remove tiny noise (e.g., < 50 pixels)
        if area < 50: 
            continue
            
        # 2. Relative Area Filter: If a fragment is too small compared to the main object (e.g., < 1%) 
        # AND it is not absolutely large enough, it is considered noise.
        # But if it is large enough (e.g., > 300 pixels), keep it even if small relative to max (might be the other half of an occluded object).
        if area < max_area * 0.01 and area < 300:
            continue

        poly = contour.flatten().astype(float)
        poly[0::2] /= W 
        poly[1::2] /= H 
        poly = np.clip(poly, 0, 1) 
        polygons.append(poly.tolist())
            
    return polygons

def polygon_to_mask(polygons, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not polygons:
        return mask.astype(bool)
    for poly in polygons:
        poly = np.array(poly).reshape(-1, 2)
        poly[:, 0] *= img_w
        poly[:, 1] *= img_h
        pts = poly.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

# --- Application State ---
sam_model = SAMWrapper(MODEL_PATH, CONFIG_PATH, device="cuda:1")
current_img_path = None
current_image = None
current_points = []
current_point_labels = []
current_masks = []
class_names = load_classes()
NULL_CLASS_LABEL = "--- Select Class ---"

# --- New Helper: Guide ---
def generate_guide_md():
    # Colors for Status UI
    c_done = "#4ade80" # Green
    c_active = "#fbbf24" # Amber
    c_pending = "#6b7280" # Gray
    c_blue = "#3b82f6" # Blue for Action
    
    # States
    is_img_sel = current_img_path is not None
    has_points = len(current_points) > 0
    has_masks = len(current_masks) > 0
    
    # --- Part A: Annotation ---
    
    # Step 1: Select Image
    if is_img_sel:
        # Hide full path, only show filename
        file_label = os.path.basename(current_img_path)
        s1 = (c_done, "✅", f"1. Ready: {file_label}")
    else:
        s1 = (c_active, "👉", "1. Select Image (Left Panel)")

    # Step 2: Pick Class & Point
    if not is_img_sel:
        s2 = (c_pending, "⚪", "2. Select Class -> Click Points")
    elif has_points:
        s2 = (c_active, "✏️", "2. Marking (Check 'Add/Remove' radio)")
    else:
        s2 = (c_active, "👉", "2. Confirm Class -> Click on Image")

    # Step 3: Add Single Mask (Blue)
    if not is_img_sel:
        s3 = (c_pending, "⚪", "3. Confirm Object")
    elif has_points:
        s3 = (c_blue, "🔵", "3. Click Blue Button [Add Mask]")
    elif has_masks:
        s3 = (c_done, "✅", "3. Object Added (Can add more)")
    else:
         s3 = (c_pending, "⚪", "3. No pending mask")

    # --- Part B: Save ---

    # Step 4: Save Image (Green)
    if not is_img_sel:
        s4 = (c_pending, "⚪", "4. Save Result")
    elif has_masks and not has_points:
        s4 = (c_done, "🟢", f"4. Done -> Click Green Button [Save Image]")
    elif has_masks:
         s4 = (c_pending, "⚪", "4. (Finish adding current object first)")
    else:
         s4 = (c_pending, "⚪", "4. Save Result (Need at least 1 mask)")

    def step_html(color, icon, text, is_bold=False):
        fw = "bold" if is_bold else "normal"
        op = "1.0" if color != c_pending else "0.5"
        return f"""
        <div style="display: flex; align-items: center; padding: 6px 10px; margin-bottom: 4px; background-color: {color}20; border-left: 4px solid {color}; border-radius: 4px; opacity: {op};">
            <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
            <span style="font-size: 0.9em; font-weight: {fw}; color: #eee;">{text}</span>
        </div>
        """
    
    html = f"""
    <div style="font-family: sans-serif; background: #1f2937; padding: 10px; border-radius: 8px; border: 1px solid #374151;">
        <div style="font-size: 1.0em; font-weight: bold; margin-bottom: 6px; color: #93c5fd; border-bottom: 1px solid #374151; padding-bottom: 4px;"> 
            Part A: Annotate Object (Loop)
        </div>
        {step_html(s1[0], s1[1], s1[2], s1[0]==c_active)}
        {step_html(s2[0], s2[1], s2[2], s2[0]==c_active)}
        {step_html(s3[0], s3[1], s3[2], s3[0]==c_blue)}
        
        <div style="font-size: 1.0em; font-weight: bold; margin-top: 12px; margin-bottom: 6px; color: #86efac; border-bottom: 1px solid #374151; padding-bottom: 4px;"> 
            Part B: Save File (Finish Image)
        </div>
        {step_html(s4[0], s4[1], s4[2], s4[0]==c_done)}
    </div>
    """
    return html

# --- Gradio Interactions ---

def update_image_list():
    files = get_image_list()
    # Key change: Use list of tuples (label, value) to hide path in UI
    choices = [(os.path.basename(f), f) for f in files]
    return gr.Dropdown(choices=choices)

def update_class_list_from_file():
    global class_names
    class_names = load_classes()
    choices = [NULL_CLASS_LABEL] + class_names
    return gr.Dropdown(choices=choices, value=NULL_CLASS_LABEL)

def select_image(filepath):
    global current_img_path, current_image, current_points, current_point_labels, current_masks
    
    if not filepath:
        return None, None, None, generate_guide_md()
    
    current_img_path = filepath
    current_image = cv2.imread(filepath)
    if current_image is not None:
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        sam_model.set_image(current_image)
    
    current_points = []
    current_point_labels = []
    
    # Load from disk
    filename = os.path.basename(filepath)
    txt_name = os.path.splitext(filename)[0] + ".txt"
    saved_txt_path = os.path.join(OUTPUT_LABELS_DIR, txt_name)
    
    if os.path.exists(saved_txt_path):
        try:
            h, w = current_image.shape[:2]
            loaded_masks = []
            with open(saved_txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        c_idx = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        mask = polygon_to_mask([coords], h, w) 
                        loaded_masks.append((c_idx, mask))
            current_masks = loaded_masks
            print(f"Loaded {len(current_masks)} masks from disk for {filename}")
        except Exception as e:
            print(f"Error loading masks from disk: {e}")
            current_masks = [] 
    else:
        current_masks = [] 
    
    vis_img = update_visualization()
    return current_image, vis_img, get_current_masks_df(), generate_guide_md(), NULL_CLASS_LABEL

def on_click_image(image, evt: gr.SelectData, point_type, selected_class):
    global current_points, current_point_labels, current_image
    
    if current_image is None:
        return image, generate_guide_md()
    
    x, y = evt.index[0], evt.index[1]
    
    label = 1 if "(+)" in point_type else 0
    current_points.append([x, y])
    current_point_labels.append(label)
    
    predicted_mask = sam_model.predict(current_points, current_point_labels)
    
    vis_img = current_image.copy()
    for c_idx, m in current_masks:
        color = (0, 255, 0) 
        # Increased opacity for existing masks in interactive view (0.5 -> 0.7)
        vis_img[m] = vis_img[m] * 0.3 + np.array(color) * 0.7
    
    if predicted_mask is not None:
        color = (255, 0, 0)
        # High opacity for current prediction (0.5 -> 0.7)
        vis_img[predicted_mask] = vis_img[predicted_mask] * 0.3 + np.array(color) * 0.7
        
    for pt, lbl in zip(current_points, current_point_labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(vis_img, tuple(pt), 5, color, -1)
        
    return vis_img, generate_guide_md()

def add_mask(selected_class):
    global current_points, current_point_labels, current_masks
    
    if selected_class == NULL_CLASS_LABEL:
        return "❌ Error: Please select a valid class first!", get_current_masks_df(), generate_guide_md(), NULL_CLASS_LABEL

    if not current_points:
        return "No active mask points", get_current_masks_df(), generate_guide_md(), selected_class

    predicted_mask = sam_model.predict(current_points, current_point_labels)
    if predicted_mask is None:
        return "Prediction Failed", get_current_masks_df(), generate_guide_md(), selected_class
        
    try:
        class_idx = class_names.index(selected_class)
    except ValueError:
        class_idx = 0
        
    current_masks.append((class_idx, predicted_mask))
    current_points = []
    current_point_labels = []
    
    return f"Added Class: {selected_class}. Total Masks: {len(current_masks)}", get_current_masks_df(), generate_guide_md(), NULL_CLASS_LABEL

def clear_points():
    global current_points, current_point_labels
    current_points = []
    current_point_labels = []
    return "Cleared All Points", generate_guide_md()

def delete_last_mask():
    global current_masks
    if current_masks:
        current_masks.pop()
        return "Removed Last Mask", get_current_masks_df(), generate_guide_md()
    return "No Mask to Remove", get_current_masks_df(), generate_guide_md()

def draw_masks_on_image(image, masks):
    vis_img = image.copy()
    for c_idx, m in masks:
        np.random.seed(c_idx)
        color = np.random.randint(0, 255, 3).tolist()
        # Increased opacity: Image * 0.3 + Color * 0.7
        vis_img[m] = vis_img[m] * 0.3 + np.array(color) * 0.7
    return vis_img

def update_visualization():
    global current_image, current_masks
    if current_image is None:
        return None
    return draw_masks_on_image(current_image, current_masks)

def get_current_masks_df():
    info = []
    for i, (c_idx, _) in enumerate(current_masks):
        c_name = class_names[c_idx] if c_idx < len(class_names) else f"Unknown({c_idx})"
        info.append([i, c_name])
    return info

def update_dataset_configs():
    try:
        categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
        with open(NOTES_JSON_PATH, "w") as f:
            json.dump({"categories": categories}, f, indent=2)
    except Exception as e:
        print(f"Error saving notes.json: {e}")

    try:
        yaml_content = f"path: {OUTPUT_DATA_DIR}\ntrain: images\nval: images\n\nnames:\n"
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        with open(DATA_YAML_PATH, "w") as f:
            f.write(yaml_content)
    except Exception as e:
        print(f"Error saving data.yaml: {e}")

def save_current_annotation():
    global current_img_path, current_masks
    if not current_img_path:
        return f"No image to save.", generate_guide_md(), NULL_CLASS_LABEL
    
    filename = os.path.basename(current_img_path)
    base_name = os.path.splitext(filename)[0]
    
    txt_path = os.path.join(OUTPUT_LABELS_DIR, base_name + ".txt")
    lines = []
    # Fixed: uses current_masks instead of undefined 'masks'
    for cls_idx, mask in current_masks:
        polys = mask_to_polygon(mask)
        for poly in polys:
            line = f"{cls_idx} " + " ".join([f"{coord:.6f}" for coord in poly])
            lines.append(line)
            
    # Always write to allow clearing annotations (writes empty file if no masks)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    
    dest_img_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
    if os.path.abspath(current_img_path) != os.path.abspath(dest_img_path):
        shutil.copy(current_img_path, dest_img_path)
        
    update_dataset_configs()
    return f"Saved: {filename} ({len(lines)} objects).", generate_guide_md(), NULL_CLASS_LABEL

def select_next_image():
    global current_img_path
    image_list = get_image_list()
    if not image_list:
        return None
    if current_img_path in image_list:
        idx = image_list.index(current_img_path)
        next_idx = (idx + 1) % len(image_list)
        new_path = image_list[next_idx]
    else:
        new_path = image_list[0]
    return new_path

# --- CSS / Layout ---
css = """
#image_display { height: 600px; }
.blue-btn { background-color: #3b82f6 !important; color: white !important; }
.green-btn { background-color: #22c55e !important; color: white !important; }
"""

with gr.Blocks(title="SAYOLO-Seg Tool", css=css) as demo:
    gr.Markdown("## SAM-Assisted YOLO-Seg Labeling Tool")
    
    with gr.Row():
        with gr.Column(scale=1):
            # File Management
            gr.Markdown("### 1. Image & Class")
            with gr.Row():
                refresh_btn = gr.Button("Refresh List")
                next_btn = gr.Button("Next Image >>")
            
            # Use tuples for choices to hide paths: (display_name, value)
            initial_files = get_image_list()
            initial_choices = [(os.path.basename(f), f) for f in initial_files]
            
            image_dropdown = gr.Dropdown(choices=initial_choices, label="Select Image")
            
            # Class Management 
            # Removed edit box, now only reading from file
            with gr.Row():
                 refresh_cls_btn = gr.Button("Reload Classes")
                 
            class_selector = gr.Dropdown(choices=[NULL_CLASS_LABEL] + class_names, value=NULL_CLASS_LABEL, label="Current Class")

            # Mask List
            gr.Markdown("### 2. Mask Manager")
            mask_df = gr.Dataframe(headers=["ID", "Class"], value=[], label="Added Masks", interactive=False)

        with gr.Column(scale=3):
            # Main Annotation Area
            original_image_state = gr.State() # Store pure image
            
            with gr.Row():
                with gr.Column(scale=4):
                    img_display = gr.Image(label="Annotation Area", type="numpy", elem_id="image_display", interactive=True)
                with gr.Column(scale=1):
                    # Guide Box
                    guide_box = gr.HTML(value=generate_guide_md(), label="Guide")
            
            with gr.Row():
                point_type = gr.Radio(["Add (+)", "Remove (-)"], value="Add (+)", label="Point Mode")
                status_info = gr.Textbox(label="Status Log", interactive=False)
            
            with gr.Row():
                with gr.Column(scale=1.1):
                    clear_pts_btn = gr.Button("Clear Points", variant="secondary")
                    undo_mask_btn = gr.Button("Undo Last Mask", variant="stop")
                
                with gr.Column(scale=1.4):
                    add_mask_btn = gr.Button("Add Mask (Finish Object)", elem_classes="blue-btn")
                    save_btn = gr.Button("Save Image", elem_classes="green-btn")

    # Event Wiring
    refresh_btn.click(update_image_list, outputs=[image_dropdown])
    refresh_cls_btn.click(update_class_list_from_file, outputs=[class_selector])
    
    # Selection
    image_dropdown.change(select_image, inputs=[image_dropdown], outputs=[original_image_state, img_display, mask_df, guide_box, class_selector])
    
    # Next button
    next_btn.click(select_next_image, outputs=[image_dropdown]) \
            .then(select_image, inputs=[image_dropdown], outputs=[original_image_state, img_display, mask_df, guide_box, class_selector])
    
    # Interaction
    img_display.select(on_click_image, inputs=[original_image_state, point_type, class_selector], outputs=[img_display, guide_box])
    
    add_mask_btn.click(add_mask, inputs=[class_selector], outputs=[status_info, mask_df, guide_box, class_selector]) \
        .then(update_visualization, outputs=[img_display])
        
    clear_pts_btn.click(clear_points, outputs=[status_info, guide_box]) \
        .then(update_visualization, outputs=[img_display])
        
    undo_mask_btn.click(delete_last_mask, outputs=[status_info, mask_df, guide_box]) \
        .then(update_visualization, outputs=[img_display])

    save_btn.click(save_current_annotation, outputs=[status_info, guide_box, class_selector])

if __name__ == "__main__":
    print("Starting Web UI (SAYOLO-Seg)...")
    print(f"Input Dir: {INPUT_IMAGES_DIR}")
    print(f"Output Dir: {OUTPUT_DATA_DIR}")
    print(f"Model: {MODEL_PATH}")
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=css)
