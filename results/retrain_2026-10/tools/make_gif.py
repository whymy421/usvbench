"""mp4 -> README GIF: real simulator frames only, sped up, resized, with a small caption bar.

python make_gif.py <in.mp4> <out.gif> --speed 3 --fps 10 --width 480 --caption "..." [--start 1.0]
"""
import argparse

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--speed", type=float, default=3.0)
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--width", type=int, default=480)
ap.add_argument("--start", type=float, default=1.0, help="skip renderer warm-up frames (seconds of video)")
ap.add_argument("--caption", default="")
a = ap.parse_args()

reader = imageio_ffmpeg.read_frames(a.src)
meta = next(reader)
w, h = meta["size"]
src_fps = meta["fps"]
stride = src_fps * a.speed / a.fps          # source frames per GIF frame
first = int(a.start * src_fps)
H = int(a.width * h / w)
bar = 22 if a.caption else 0
try:
    font = ImageFont.truetype("segoeui.ttf", 13)
except OSError:
    font = ImageFont.load_default()
frames, nxt = [], float(first)
for i, raw in enumerate(reader):
    if i < nxt:
        continue
    nxt += stride
    im = Image.frombytes("RGB", (w, h), raw).resize((a.width, H), Image.LANCZOS)
    if bar:
        canvas = Image.new("RGB", (a.width, H + bar), (252, 252, 251))
        canvas.paste(im, (0, 0))
        t = (i - first) / src_fps
        ImageDraw.Draw(canvas).text((6, H + 3), f"{a.caption}   t = {t + a.start:4.1f} s   ({a.speed:g}x)",
                                    fill=(82, 81, 78), font=font)
        im = canvas
    frames.append(im.quantize(colors=int(__import__("os").environ.get("GIF_COLORS","128")), method=Image.Quantize.MEDIANCUT, dither=(Image.Dither.FLOYDSTEINBERG if __import__("os").environ.get("GIF_DITHER") else Image.Dither.NONE)))
reader.close()
frames[0].save(a.dst, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
import os
print(f"{a.dst}: {len(frames)} frames, {os.path.getsize(a.dst)/1e6:.2f} MB")
