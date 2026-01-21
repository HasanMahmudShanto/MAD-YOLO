import sys
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Hardware Check
    print("🔍 Checking hardware...")
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"✅ Found XPU: {torch.xpu.get_device_name(0)}")
        target_device = 'xpu'
    else:
        print("❌ No XPU found. Exiting.")
        sys.exit(1)

    # 2. Load Model
    # We load your custom architecture
    model = YOLO("yolov8_custom_final.yaml") 

    # 3. START TRAINING
    print("🔥 Starting training...")
    model.train(
        # 👇 UPDATE: This now points to your newly created dataset
        data="D:/research/tt100k_2021/tt100k_2021/tt100k_paper_exact/data.yaml",
        
        epochs=100,
        imgsz=640,
        batch=16,
        device=target_device, 
        project="tt100k_research",
        name="b580_paper_exact_run",
        workers=0,
        amp=False
    )