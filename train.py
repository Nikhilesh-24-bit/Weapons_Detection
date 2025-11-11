from ultralytics import YOLO
import os

# Set PyTorch CUDA memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better CUDA error debugging

if __name__ == '__main__':
    # Load model
    model = YOLO('yolov8n.pt')  # Nano model for low memory usage

    # Train
    model.train(
        data=r'C:\Users\nikhi\Desktop\weapon-detection\data.yaml',
        epochs=50,
        imgsz=416,  # Ensure lower resolution
        batch=4,   # Further reduced to minimize VRAM usage
        workers=4,  # Reduced to lower memory overhead
        device=0   # Use GPU
    )