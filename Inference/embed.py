import os
import sys

import cv2
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(CUR_DIR, "runtime")
if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from runtime.data.utils import encode

INPUT_DIR = os.path.join(CUR_DIR, "input_images")
OUTPUT_DIR = os.path.join(CUR_DIR, "embedded_output")
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
WATERMARK_TXT = os.path.join(OUTPUT_DIR, "watermark.txt")

ALPHA = 8
BETA = 96
MSG_INFO_LEN = 64
MSG_ENCODED_LEN = 240
BLOCK_SIZE = 12
BLOCK_NUM = 16
MARKSHAPE_PARA = 5

# Optional simulation setting, disabled by default.
ENABLE_SCREENSHOT_SIM = False
OUTPUT_H = 1024
OUTPUT_W = 1024


# def simulate_screenshot(input_image, subimage_size, set_image_w=0, set_image_h=0):
#     height, width, _ = input_image.shape
#     if set_image_h and set_image_w:
#         image_h = set_image_h
#         image_w = set_image_w
#     else:
#         image_h = np.random.randint(2 * subimage_size, max(height, 2 * subimage_size + 1))
#         image_w = np.random.randint(2 * subimage_size, max(width, 2 * subimage_size + 1))

#     image_h = min(int(image_h), int(height))
#     image_w = min(int(image_w), int(width))
#     max_h = height - image_h
#     max_w = width - image_w
#     rnd_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
#     rnd_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
#     return input_image[rnd_h:rnd_h + image_h, rnd_w:rnd_w + image_w, :]


def generate_random_watermark(info_len=64, encoded_len=240):
    payload = [np.random.randint(0, 2) for _ in range(info_len)]
    watermark = encode(payload, unit_info_bits=info_len, unit_total_bits=encoded_len)
    if len(watermark) != encoded_len:
        raise ValueError(f"encoded watermark length mismatch: {len(watermark)} != {encoded_len}")
    return watermark


def insert_and_resize(watermark_bch, size=(16, 16)):
    new_watermark = [0] * (size[0] * size[1])
    insert_indices = [0, 1, 16, 17, 14, 15, 30, 31, 224, 225, 240, 241, 238, 239, 254, 255]
    for i in insert_indices:
        new_watermark[i] = 0

    watermark_iter = iter(watermark_bch)
    for i in range(len(new_watermark)):
        if i not in insert_indices:
            new_watermark[i] = next(watermark_iter)

    return np.array(new_watermark).reshape(size)


def build_pm(block_size, markshape_para):
    dxy = lambda x, y: ((x - (block_size // 2)) ** 2 + (y - block_size // 2) ** 2) ** 0.5
    pplus = np.zeros((block_size, block_size, 1), dtype=np.int32)
    pminus = np.zeros((block_size, block_size, 1), dtype=np.int32)
    for x in range(block_size):
        for y in range(block_size):
            pplus[x, y, 0] = dxy(x, y) / markshape_para
            pminus[x, y, 0] = -dxy(x, y) / markshape_para
    return pplus, pminus


def build_bm(msg_matrix, block_size, block_num, markshape_para):
    pplus, pminus = build_pm(block_size, markshape_para)

    locate_1 = np.copy(pminus)
    locate_2 = np.copy(pminus)
    locate_3 = np.copy(pminus)
    locate_4 = np.copy(pminus)
    for i in range(block_size):
        for j in range(block_size):
            if i < 6 and j < 6:
                locate_1[i, j, 0] = -pminus[i, j, 0]
            if i < 6 and j > 5:
                locate_2[i, j, 0] = -pminus[i, j, 0]
            if i > 5 and j < 6:
                locate_3[i, j, 0] = -pminus[i, j, 0]
            if i > 5 and j > 5:
                locate_4[i, j, 0] = -pminus[i, j, 0]

    def build_row(is_blue, i):
        row = []
        for j in range(block_num):
            pixel_index = i * block_num + j
            if pixel_index in [0, 14, 224, 238]:
                block = locate_4
            elif pixel_index in [1, 15, 225, 239]:
                block = locate_3
            elif pixel_index in [16, 30, 240, 254]:
                block = locate_2
            elif pixel_index in [17, 31, 241, 255]:
                block = locate_1
            else:
                if is_blue:
                    block = pminus if msg_matrix[i][j] else pplus
                else:
                    block = pplus if msg_matrix[i][j] else pminus
            row.append(block)
        return np.hstack(row)

    bplus_r = np.vstack([build_row(False, i) for i in range(block_num)])
    bplus_b = np.vstack([build_row(True, i) for i in range(block_num)])
    return bplus_b, bplus_r


def build_image(watermark, rawimg, block_size, block_num, markshape_para, alpha, beta, subimage_size):
    rawimg = rawimg.astype(np.uint32)
    watermark_256 = insert_and_resize(watermark)
    bplus_b, bplus_r = build_bm(watermark_256, block_size, block_num, markshape_para)

    ib = rawimg[:, :, 0].reshape(subimage_size, subimage_size, 1)
    ig = rawimg[:, :, 1].reshape(subimage_size, subimage_size, 1)
    ir = rawimg[:, :, 2].reshape(subimage_size, subimage_size, 1)

    mask_b = 128 + beta * bplus_b
    mask_g = np.ones_like(mask_b) * 128
    mask_r = 128 + beta * bplus_r

    ib1 = (alpha * mask_b + (255 - alpha) * ib) / 255.0
    ig1 = (alpha * mask_g + (255 - alpha) * ig) / 255.0
    ir1 = (alpha * mask_r + (255 - alpha) * ir) / 255.0

    ib1 = np.clip(ib1, 0, 255).astype(np.uint8)
    ig1 = np.clip(ig1, 0, 255).astype(np.uint8)
    ir1 = np.clip(ir1, 0, 255).astype(np.uint8)
    return np.dstack((ib1, ig1, ir1)).astype(np.uint8)


def watermark_embedding(rawimg, watermark, block_size, block_num, markshape_para, alpha, beta, subimage_size):
    height_num = rawimg.shape[0] // subimage_size
    width_num = rawimg.shape[1] // subimage_size
    embedded_image = []
    for i in range(height_num + 1):
        row = []
        for j in range(width_num + 1):
            subimage = rawimg[i * subimage_size:(i + 1) * subimage_size, j * subimage_size:(j + 1) * subimage_size]
            if subimage.shape[0] < subimage_size or subimage.shape[1] < subimage_size:
                h_padding = max(0, subimage_size - subimage.shape[0])
                w_padding = max(0, subimage_size - subimage.shape[1])
                subimage = np.pad(subimage, ((0, h_padding), (0, w_padding), (0, 0)), mode="constant")
            row.append(build_image(watermark, subimage, block_size, block_num, markshape_para, alpha, beta, subimage_size))
        embedded_image.append(np.hstack(row))
    result_image = np.vstack(embedded_image)
    return result_image[0:rawimg.shape[0], 0:rawimg.shape[1], :].astype(np.uint8)


def run_embed(wm_mode="64"):
    if wm_mode not in {"64", "240"}:
        raise ValueError("wm_mode must be '64' or '240'")

    subimage_size = BLOCK_SIZE * BLOCK_NUM
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    for fn in os.listdir(IMAGE_OUTPUT_DIR):
        if fn.lower().endswith(".png"):
            os.remove(os.path.join(IMAGE_OUTPUT_DIR, fn))

    image_list = [x for x in os.listdir(INPUT_DIR) if x.lower().endswith(".png")]
    image_list.sort()
    if not image_list:
        raise ValueError(f"No png files found in {INPUT_DIR}")

    perm = np.random.RandomState(42).permutation(MSG_ENCODED_LEN).tolist()
    inverse_perm = [0] * len(perm)
    for shuffled_idx, original_idx in enumerate(perm):
        inverse_perm[original_idx] = shuffled_idx
    embed_indices = inverse_perm if wm_mode == "240" else None

    with open(WATERMARK_TXT, "w") as wf:
        for idx, image_name in enumerate(image_list, start=1):
            image_path = os.path.join(INPUT_DIR, image_name)
            rawimg = cv2.imread(image_path)
            if rawimg is None:
                continue

            payload_64 = [np.random.randint(0, 2) for _ in range(MSG_INFO_LEN)]
            watermark_240 = encode(payload_64, unit_info_bits=MSG_INFO_LEN, unit_total_bits=MSG_ENCODED_LEN)
            if len(watermark_240) != MSG_ENCODED_LEN:
                raise ValueError(f"encoded watermark length mismatch: {len(watermark_240)} != {MSG_ENCODED_LEN}")

            saved_bits_240 = [watermark_240[i] for i in perm]
            embedded_bits = saved_bits_240 if embed_indices is None else [saved_bits_240[i] for i in embed_indices]
            saved_bits = payload_64 if wm_mode == "64" else saved_bits_240

            result_image = watermark_embedding(
                rawimg, embedded_bits, BLOCK_SIZE, BLOCK_NUM, MARKSHAPE_PARA, ALPHA, BETA, subimage_size
            )

            # # Optional and disabled by default.
            # if ENABLE_SCREENSHOT_SIM:
            #     result_image = simulate_screenshot(result_image, subimage_size, OUTPUT_H, OUTPUT_W)

            out_name = f"{idx}.png"
            cv2.imwrite(os.path.join(IMAGE_OUTPUT_DIR, out_name), result_image)
            wf.write(f"{idx}-" + "".join(map(str, saved_bits)) + "\n")

    print(f"[embed] mode={wm_mode}, images={len(image_list)}")
    print(f"[embed] images -> {IMAGE_OUTPUT_DIR}")
    print(f"[embed] watermark -> {WATERMARK_TXT}")


if __name__ == "__main__":
    run_embed("64")
