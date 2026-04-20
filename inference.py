import os
import glob
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

import utils
from model_arch import ResUNet, RefineUNet


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(stage1_path, stage2_path, device):
    print("Loading models...")
    model1 = ResUNet(in_ch=48, out_ch=3).to(device)
    model2 = RefineUNet().to(device)

    model1.load_state_dict(torch.load(stage1_path, map_location=device))
    model2.load_state_dict(torch.load(stage2_path, map_location=device))

    model1.eval()
    model2.eval()
    return model1, model2


def find_npy_files(input_dir):
    pattern = os.path.join(input_dir, "**", "*.npy")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def save_output_image(output_tensor, save_path):
    out_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = np.clip(out_img, 0, 1)
    out_img = (out_img * 255.0).astype(np.uint8)

    # model output is RGB, cv2 expects BGR
    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ok = cv2.imwrite(save_path, out_img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image: {save_path}")


def run_inference(input_dir, output_dir, stage1_path, stage2_path, keep_structure=True):
    device = get_device()

    print("Starting inference...")
    print(f"   Input dir:   {input_dir}")
    print(f"   Output dir:  {output_dir}")
    print(f"   Stage1:      {stage1_path}")
    print(f"   Stage2:      {stage2_path}")
    print(f"   Device:      {device}")
    print(f"   Keep struct: {keep_structure}")

    os.makedirs(output_dir, exist_ok=True)

    model1, model2 = load_models(stage1_path, stage2_path, device)

    files = find_npy_files(input_dir)
    if not files:
        print("No .npy files found.")
        return

    print(f"Found {len(files)} .npy files")

    for f_path in tqdm(files, desc="Processing"):
        try:
            raw = np.load(f_path)
            inp = utils.preprocess_raw_data(raw, device=device)

            if inp is None:
                print(f"Skipping invalid file: {f_path}")
                continue

            with torch.no_grad():
                coarse = model1(inp)
                fine = model2(coarse).clamp(0, 1)

            base_name = os.path.splitext(os.path.basename(f_path))[0] + ".png"

            if keep_structure:
                rel_dir = os.path.relpath(os.path.dirname(f_path), input_dir)
                if rel_dir == ".":
                    save_path = os.path.join(output_dir, base_name)
                else:
                    save_path = os.path.join(output_dir, rel_dir, base_name)
            else:
                save_path = os.path.join(output_dir, base_name)

            save_output_image(fine, save_path)

        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    print("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPAD image reconstruction inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save reconstructed .png files")
    parser.add_argument("--stage1", type=str, required=True, help="Path to stage1_coarse_ema.pth")
    parser.add_argument("--stage2", type=str, required=True, help="Path to stage2_refine_ema.pth")
    parser.add_argument("--flat", action="store_true", help="Do not keep subfolder structure")
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stage1_path=args.stage1,
        stage2_path=args.stage2,
        keep_structure=not args.flat,
    )
