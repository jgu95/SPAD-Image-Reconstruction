import numpy as np
import cv2
import torch

def preprocess_raw_data(raw, device="cuda"):
    """
    Converts raw SPAD data to Stage 1 input Tensor.
    Handles both raw compressed data (1024, H, W_compressed) 
    and pre-processed feature data (48, H, W).
    """
    
    # 1. Compatibility Check: If already 48-channel feature map
    if raw.ndim == 3 and raw.shape[0] == 48:
        return torch.from_numpy(raw.astype(np.float32)).unsqueeze(0).to(device)

    # 2. Compatibility Fix (3D -> 4D)
    # If raw is (1024, 800, 100), expand to include channel dim then repeat
    if raw.ndim == 3: 
        raw = np.expand_dims(raw, axis=-1)
        raw = np.repeat(raw, 3, axis=-1)
    
    # Ensure correct time dimension
    if raw.shape[0] != 1024: 
        return None

    # 3. Slicing and Unpacking
    # Take the last 192 time bins
    raw_last = raw[1024-192:]
    proxies = []
    chunk_len = 192 // 16
    
    for i in range(16):
        block = raw_last[i*chunk_len : (i+1)*chunk_len]
        # np.unpackbits expands the width (e.g., 100 -> 800)
        bits = np.unpackbits(block, axis=2) 
        proxies.append(bits.mean(axis=0).astype(np.float32))
    
    # Shape: (16, H, W, 3)
    proxies = np.stack(proxies, axis=0) 

    # 4. Enhancement (Log + Gamma + GaussianBlur)
    enh = []
    for p in proxies:
        img = p.copy()
        if img.max() > 1.0 + 1e-4: img /= 255.0
        
        # Robust normalization
        perc = np.percentile(img, 98)
        if perc > 0: img /= perc
        img = np.clip(img, 0, 1)
        
        # Log and Gamma correction
        img = np.log1p(img)
        img /= (img.max() + 1e-6)
        img = img ** 0.5 
        
        # Denoising logic (Preserving Feature Alignment)
        res = np.zeros_like(img)
        for c in range(3): 
            res[...,c] = cv2.GaussianBlur(img[...,c], (3,3), 0.6)
        enh.append(np.clip(res, 0, 1))
        
    enh = np.stack(enh, axis=0)
    
    # 5. Convert to Tensor
    # enh shape is (16, H, W, 3).
    # We transpose to (16, 3, H, W) then flatten the first two dims to 48.
    # Note: We use enh.shape for width/height to automatically handle the unpacked width.
    feat = enh.transpose(0, 3, 1, 2).reshape(48, enh.shape[1], enh.shape[2])
    
    tensor = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
    
    return tensor
