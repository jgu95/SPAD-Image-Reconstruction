import os
import glob
import torch
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# Import skeleton and utils
import utils
from model_arch import ResUNet, RefineUNet

# ================= Configuration =================
INPUT_FOLDER = "/content/trainset5"       # Input directory
OUTPUT_FOLDER = "/content/final_results_5" # Output directory

DEFAULT_STAGE1 = "stage1_coarse_ema.pth"
DEFAULT_STAGE2 = "stage2_refine_ema.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESUME_MODE = False

def run_inference(input_dir, output_dir, stage1_path, stage2_path):
    print(f"🚀 Starting Inference...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Device: {DEVICE}")

    # Setup directories
    if not RESUME_MODE and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load Models
    print("⏳ Loading models...")
    model1 = ResUNet(in_ch=48, out_ch=3).to(DEVICE)
    model2 = RefineUNet().to(DEVICE)
    
    try:
        model1.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
        model2.load_state_dict(torch.load(stage2_path, map_location=DEVICE))
        model1.eval()
        model2.eval()
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # Find files
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files: 
        print("❌ No .npy files found.")
        return

    print(f"📸 Pending processing: {len(files)} files")

    # Processing Loop
    for f_path in tqdm(files):
        base_name = os.path.basename(f_path).replace(".npy", "")
        save_path = os.path.join(output_dir, f"{base_name}.png")

        # Skip if in resume mode and file exists
        if RESUME_MODE and os.path.exists(save_path): 
            continue

        # Copy Ground Truth (GT) if available for comparison
        gt_src = os.path.join(input_dir, f"{base_name}.png")
        if os.path.exists(gt_src):
            shutil.copy(gt_src, os.path.join(output_dir, f"{base_name}_gt.png"))

        try:
            # Load raw data
            raw = np.load(f_path)
            
            # Preprocess (Logic is now centralized in utils.py)
            inp = utils.preprocess_raw_data(raw, device=DEVICE)

            if inp is None:
                print(f"⚠️ Skipping invalid file: {base_name}")
                continue

            # Inference
            with torch.no_grad():
                coarse = model1(inp).clamp(0, 1)
                fine = model2(coarse).clamp(0, 1)

            # Post-process and Save
            out_img = fine.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = (out_img * 255.0).astype(np.uint8)
            
            # Convert RGB (Model output) to BGR (OpenCV expectation)
            cv2.imwrite(save_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"⚠️ Error processing {base_name}: {e}")

    print(f"✅ Inference complete!")

if __name__ == "__main__":
    run_inference(INPUT_FOLDER, OUTPUT_FOLDER, DEFAULT_STAGE1, DEFAULT_STAGE2)
