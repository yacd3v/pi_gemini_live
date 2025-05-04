import spidev, RPi.GPIO as GPIO, time
from PIL import Image, ImageDraw

# --- pins ---
DC, RST, BL = 25, 27, 18     # as wired

# --- setup GPIO ---
GPIO.setmode(GPIO.BCM)
for p in (DC, RST, BL):
    GPIO.setup(p, GPIO.OUT)
GPIO.output(BL, 1)

# --- hardware reset ---
GPIO.output(RST, 0); time.sleep(0.1); GPIO.output(RST, 1)

# --- open SPI ---
spi = spidev.SpiDev(0, 0)          # bus 0, CE0
spi.mode = 0                       # CPOL=0, CPHA=0
spi.max_speed_hz = 62_500_000      # 62.5 MHz runs fine on Pi 5

def cmd(c, data=None):
    GPIO.output(DC, 0); spi.xfer([c])
    if data:
        GPIO.output(DC, 1); spi.xfer(data)

# GC9A01 init (minimal)
cmd(0x36, [0x00])      # MADCTL
cmd(0x3A, [0x05])      # 16-bit colour
cmd(0x11); time.sleep(0.12)
cmd(0x29)              # display ON

# draw a red circle in RAM then send
img = Image.new("RGB", (240, 240), "black")
d   = ImageDraw.Draw(img)
d.ellipse((20, 20, 220, 220), outline="red", width=6)
buf = img.tobytes()

# window = full screen
cmd(0x2A, [0,0,0,239]) # column
cmd(0x2B, [0,0,0,239]) # row
cmd(0x2C)              # RAMWR
GPIO.output(DC, 1)

# Send data in chunks of 4096 bytes
chunk_size = 4096
for i in range(0, len(buf), chunk_size):
    chunk = buf[i:i + chunk_size]
    spi.writebytes(list(chunk))

time.sleep(5)
spi.close(); GPIO.cleanup()