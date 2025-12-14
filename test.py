# text2img_gray.py
import multiprocessing as mp
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = 256
PADDING = 0
BG = 255         # white in "L"
FG = 0           # black in "L"
LINE_SPACING = 1.15  # line height multiplier

# Try a common TTF; fall back to PIL's bitmap default if not found.
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    ImageFont.truetype(DEFAULT_FONT_PATH, 12)
except Exception:
    print("Failed to load font")
    DEFAULT_FONT_PATH = None

@lru_cache(maxsize=256)
def _font(font_path, size):
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()

def _wrap_lines(text, draw, font, max_width):
    lines = []
    for paragraph in (text.splitlines() or [""]):
        words, line = paragraph.split(" "), ""
        for w in words:
            candidate = w if not line else f"{line} {w}"
            if draw.textlength(candidate, font=font) <= max_width:
                line = candidate
            else:
                if not line:
                    # break very long “word” (e.g., URL) by characters with hyphen
                    accum = ""
                    for ch in w:
                        if draw.textlength(accum + ch, font=font) <= max_width:
                            accum += ch
                        else:
                            lines.append(accum[:-1] + "-")
                            accum = accum[-1:] + ch
                    line = accum
                else:
                    lines.append(line)
                    line = w
        if line:
            lines.append(line)
    return lines

def _measure_block(lines, draw, font):
    ascent, descent = font.getmetrics()
    base_line = ascent + descent
    line_h = int(base_line * LINE_SPACING)
    width = 0
    for ln in lines:
        width = max(width, draw.textlength(ln, font=font))
    height = line_h * (len(lines) if lines else 1)
    return width, height, line_h

def _fit(text, font_path=DEFAULT_FONT_PATH, box=CANVAS, padding=PADDING):
    max_w = max_h = box - 2 * padding
    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)

    lo, hi, best = 1, 240, None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _font(font_path, mid)
        lines = _wrap_lines(text, draw, font, max_w)
        w, h, line_h = _measure_block(lines, draw, font)
        if w <= max_w and h <= max_h:
            best = (mid, lines, w, h, line_h)
            lo = mid + 1
        else:
            hi = mid - 1

    if not best:
        # fallback to smallest size, trim with ellipsis if still too tall
        font = _font(font_path, 6)
        lines = _wrap_lines(text, draw, font, max_w)
        _, h, line_h = _measure_block(lines, draw, font)
        max_lines = max(1, (max_h // line_h))
        lines = lines[:max_lines]
        if lines:
            # ensure last line fits width with ellipsis if needed
            last = lines[-1]
            while last and draw.textlength(last + "…", font=font) > max_w:
                last = last[:-1]
            lines[-1] = (last + "…") if last else "…"
        return font, lines

    size, lines, *_ = best
    return _font(font_path, size), lines

def render_text_image(
    text,
    out_path=None,
    out_type="numpy", # PIL or numpy
    font_path=DEFAULT_FONT_PATH,
    box=CANVAS,
    padding=PADDING,
    align="left",      # left|center|right
    valign="middle",     # top|middle|bottom
    mode="L",            # "L" (8-bit 1-channel) output; use "1" for 1-bit after conversion
):
    # Draw on 8-bit grayscale for antialiased text, then optionally convert to 1-bit.
    imgL = Image.new("L", (box, box), BG)
    draw = ImageDraw.Draw(imgL)
    font, lines = _fit(text, font_path=font_path, box=box, padding=padding)

    _, h, line_h = _measure_block(lines, draw, font)
    if valign == "top":
        y = padding
    elif valign == "bottom":
        y = box - padding - h
    else:
        y = (box - h) // 2

    for ln in lines:
        w = draw.textlength(ln, font=font)
        if align == "left":
            x = padding
        elif align == "right":
            x = box - padding - int(w)
        else:
            x = (box - int(w)) // 2
        draw.text((x, y), ln, font=font, fill=FG)
        y += line_h

    img_out = imgL if mode == "L" else imgL.convert("1")  # 1-bit if requested
    if out_path:
        img_out.save(out_path, format="PNG")
    if out_type == "PIL":
        return img_out
    elif out_type == "numpy":
        return np.asarray(img_out)
    else:
        raise ValueError(f"Invalid out_type: {out_type}")


if __name__ == "__main__":
    import time
    from multiprocessing import get_context

    import jax
    from tqdm import tqdm

    n_workers = 10
    batch_size = 10000
    texts = [f"Your paragraph here. This is the text number {i} and we need to render it the best we can. However it may not be the best we can do. Let's see what happens from there." for i in range(batch_size)]

    #print(f"Jax devices: {jax.device_count()}")

    ctx = get_context("spawn")

    with ctx.Pool(n_workers) as p:
        start_time = time.time()
        results = list(tqdm(p.imap(render_text_image, texts), total=len(texts)))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

    print(jax.device_count())
