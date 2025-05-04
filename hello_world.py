#!/usr/bin/python
# coding: utf-8
"""
Improved eye animation for 1.28‑inch round LCD on Raspberry Pi 5.
Focus: smoother lids, blinks, gaze shifts and a few emotions.
The code is compact so you can tweak timings and sizes easily.
"""
import sys, time, random, math
from pathlib import Path
from PIL import Image, ImageDraw
import spidev as SPI
sys.path.append("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/..")
from lib import LCD_1inch28


class EyeRenderer:
    """Draw a single eye frame."""

    def __init__(self, disp):
        self.disp = disp
        self.w, self.h = disp.width, disp.height
        self.cx, self.cy = self.w // 2, self.h // 2
        self.eye_r = min(self.w, self.h) // 3
        self.pupil_r = self.eye_r // 3

    # ---------------- helper maths -----------------
    @staticmethod
    def ease(t: float) -> float:
        """Sinusoidal ease in/out. t∈[0,1]."""
        return 0.5 - 0.5 * math.cos(math.pi * t)

    # ---------------- drawing primitives ----------
    def _draw_heart(self, d: ImageDraw.ImageDraw, center, size):
        x, y = center
        pts = [
            (x, y + size),
            (x - size, y),
            (x - size // 2, y - size // 2),
            (x, y - size // 4),
            (x + size // 2, y - size // 2),
            (x + size, y),
        ]
        d.polygon(pts, fill="RED")

    def frame(self, openness: float = 1.0, gaze=(0.0, 0.0), style="pupil") -> Image.Image:
        """
        openness 0 (closed) → 1 (open).
        gaze is ±1 in x/y.
        style: "pupil" or "heart".
        """
        img = Image.new("RGB", (self.w, self.h), "BLACK")
        d = ImageDraw.Draw(img)

        # Eyeball
        d.ellipse(
            (
                self.cx - self.eye_r,
                self.cy - self.eye_r,
                self.cx + self.eye_r,
                self.cy + self.eye_r,
            ),
            fill="WHITE",
        )

        # Pupil / heart
        px = self.cx + int(gaze[0] * self.eye_r * 0.45)
        py = self.cy + int(gaze[1] * self.eye_r * 0.45)
        if style == "heart":
            self._draw_heart(d, (px, py), self.pupil_r)
        else:
            d.ellipse(
                (
                    px - self.pupil_r,
                    py - self.pupil_r,
                    px + self.pupil_r,
                    py + self.pupil_r,
                ),
                fill="BLACK",
            )

        # Eyelids (simple rectangles – cheap & fast)
        lid_h = int((1.0 - openness) * self.eye_r * 2)
        if lid_h:
            # top lid
            d.rectangle((0, self.cy - self.eye_r, self.w, self.cy - self.eye_r + lid_h // 2), fill="BLACK")
            # bottom lid
            d.rectangle((0, self.cy + self.eye_r - lid_h // 2, self.w, self.cy + self.eye_r), fill="BLACK")

        return img

    # --------------- canned sequences -------------
    def blink(self, fps=25):
        frames = []
        steps = int(fps * 0.15)  # 150 ms close + 150 ms open
        for i in range(steps):
            t = i / (steps - 1)
            if t <= 0.5:
                op = 1 - self.ease(t * 2)
            else:
                op = self.ease((t - 0.5) * 2)
            frames.append(self.frame(openness=op))
        return frames

    def look(self, dx, dy, dur=0.4, fps=25):
        steps = int(fps * dur)
        return [
            self.frame(gaze=(self.ease(i / (steps - 1)) * dx, self.ease(i / (steps - 1)) * dy))
            for i in range(steps)
        ]

    def emotion(self, name: str, fps=25):
        if name == "happy":
            return self.look(0.6, -0.3, dur=0.8, fps=fps)
        if name == "angry":
            seq = self.look(-0.5, -0.4, dur=0.5, fps=fps)
            # add quick half‑blink for scowl
            mid = len(seq) // 2
            seq.insert(mid, self.frame(openness=0.3, gaze=(-0.5, -0.4)))
            return seq
        if name == "loving":
            return [self.frame(style="heart") for _ in range(int(fps * 0.6))]
        return [self.frame() for _ in range(int(fps * 0.6))]


# ------------------------- main loop ---------------------------

def main():
    disp = LCD_1inch28.LCD_1inch28()
    disp.Init()
    disp.clear()
    disp.bl_DutyCycle(60)  # a bit brighter

    eye = EyeRenderer(disp)
    fps = 25

    try:
        while True:
            # natural idle: random gaze shift + blink
            for frame in eye.look(random.uniform(-1, 1), random.uniform(-1, 1), dur=0.6, fps=fps):
                disp.ShowImage(frame.rotate(180))
                time.sleep(1 / fps)
            if random.random() < 0.4:
                for frame in eye.blink(fps=fps):
                    disp.ShowImage(frame.rotate(180))
                    time.sleep(1 / fps)
            # play one emotion occasionally
            if random.random() < 0.2:
                emo = random.choice(["happy", "angry", "loving"])
                for frame in eye.emotion(emo, fps=fps):
                    disp.ShowImage(frame.rotate(180))
                    time.sleep(1 / fps)
    except KeyboardInterrupt:
        # Graceful exit
        disp.bl_DutyCycle(0)
        disp.module_exit()


if __name__ == "__main__":
    main()
