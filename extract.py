import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from statistics import StatisticsError
from tqdm import tqdm


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(CUR_DIR, "runtime")
if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from runtime.analysis_lgz import cal_contour
from runtime.attack.JpegCompression import JpegFASL
from runtime.data.utils import decode
from runtime.models.Pipeline import TwoStage

INPUT_DIR = os.path.join(CUR_DIR, "embedded_output", "images")
TO_EXTRACT_DIR = os.path.join(CUR_DIR, "to_extract")
OUTPUT_TXT = os.path.join(CUR_DIR, "extract_output", "extracted_watermark.txt")
CKPT_DIR = os.path.join(CUR_DIR, "runtime", "Save_data", "twostage", "screen_watermark_weights", "a8_b96")
DEVICE = "cuda"
VOTING_TIMES = 25
SUBIMAGE_SIZE = 192

# Optional attack setting, disabled by default.
# USE_JPEG_ATTACK = False
# JPEG_QUALITY = 95


def find_max_ckpt(ckpt_path):
    ckpt_list = [x for x in os.listdir(ckpt_path) if x != "lightning_logs"]
    ckpt_mod_list = [i[-9:-5] for i in ckpt_list]
    idx = ckpt_mod_list.index(max(ckpt_mod_list))
    return ckpt_list[idx]


def generate_patch(input_image, patch_size):
    height, width, _ = input_image.shape
    patch_size = int(min(patch_size, height, width))
    max_h = height - patch_size
    max_w = width - patch_size
    lap_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
    lap_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
    return input_image[lap_h: lap_h + patch_size, lap_w:lap_w + patch_size, :]

def run_extract(wm_mode="64", source="embedded"):
    if wm_mode not in {"64", "240"}:
        raise ValueError("wm_mode must be '64' or '240'")

    np2tensor = lambda x: torch.from_numpy(np.array(x).astype(np.float32)).permute((2, 0, 1))
    pad_loc = [0, 1, 14, 15, 16, 17, 30, 31, 224, 225, 238, 239, 240, 241, 254, 255]

    ckpt_path = os.path.join(CKPT_DIR, find_max_ckpt(CKPT_DIR))
    perm = np.random.RandomState(42).permutation(240).tolist()
    inverse_perm = [0] * len(perm)
    for shuffled_idx, original_idx in enumerate(perm):
        inverse_perm[original_idx] = shuffled_idx
    to_64 = (wm_mode == "64")
    reorder_indices = inverse_perm if to_64 else perm
    zero_len = 64 if to_64 else 240

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")
    model = TwoStage(out_dir="./", secret_hw=16, block_size=192, tile_size=12, do_parser=True)
    weights = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(weights["state_dict"])
    model = model.to(device)
    model.eval()

    jpeg_layer = JpegFASL().to(device).eval()

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    os.makedirs(TO_EXTRACT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    active_input_dir = INPUT_DIR if source == "embedded" else TO_EXTRACT_DIR
    image_list = sorted([x for x in os.listdir(active_input_dir) if x.lower().endswith(".png")])
    if not image_list:
        raise ValueError(f"No png files found in {active_input_dir}")

    with open(OUTPUT_TXT, "w") as out_f:
        for image_name in tqdm(image_list):
            base_name = image_name[:-4]

            image_path = os.path.join(active_input_dir, image_name)
            image_data = cv2.imread(image_path)
            if image_data is None:
                continue
            image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB) / 255

            input_batches = []
            for _ in range(VOTING_TIMES):
                cropped_image = generate_patch(image, patch_size=2 * SUBIMAGE_SIZE)
                input_batches.append(np2tensor(cropped_image).unsqueeze(0))
            cropped_images = torch.cat(input_batches, dim=0).to(device)

            with torch.no_grad():
                # if USE_JPEG_ATTACK:
                #     cropped_images, _, _ = jpeg_layer(cropped_images, JPEG_QUALITY)

                pred_loc = model.locator(cropped_images)
                grids_pred = (pred_loc.permute((0, 2, 3, 1)) * 255).cpu().numpy().astype(np.uint8)
                batch_images = []
                for image_index in range(len(grids_pred)):
                    tmp_mask = grids_pred[image_index]
                    tmp_input_image = cropped_images[image_index]
                    _, tmp_mask = cv2.threshold(tmp_mask, 127, 255, cv2.THRESH_BINARY)
                    try:
                        coords = cal_contour(tmp_mask)
                        start_point = coords[0]
                        end_point = coords[2]
                        aligned_image = tmp_input_image[
                            :, start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1
                        ]
                    except StatisticsError:
                        aligned_image = None
                    if aligned_image is None or aligned_image.shape[-1] == 0 or aligned_image.shape[-2] == 0:
                        continue
                    aligned_image = F.interpolate(aligned_image.unsqueeze(0), (192, 192))
                    batch_images.append(aligned_image)

                if not batch_images:
                    pred_bits = [0] * zero_len
                    out_f.write(f"{base_name}-" + "".join(map(str, pred_bits)) + "\n")
                    continue

                batch_images = torch.cat(batch_images, dim=0)
                extracted = model(batch_images)
                pred = torch.round(torch.sigmoid(extracted))
                pred = torch.round(torch.mean(pred, dim=0))

                pred_secret = pred.flatten(1).cpu().numpy().astype(np.int8).tolist()
                for loc in pad_loc[::-1]:
                    pred_secret[0].pop(loc)

                pred_240 = pred_secret[0]
                reordered = [pred_240[idx] for idx in reorder_indices]
                pred_bits = decode(reordered, unit_info_bits=64, unit_total_bits=240) if to_64 else reordered
                out_f.write(f"{base_name}-" + "".join(map(str, pred_bits)) + "\n")

    print(f"[extract] mode={wm_mode}, images={len(image_list)}")
    print(f"[extract] output -> {OUTPUT_TXT}")


if __name__ == "__main__":
    run_extract("64")
