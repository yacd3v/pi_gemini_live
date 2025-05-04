#!/usr/bin/env python3
import smbus2, time
bus   = smbus2.SMBus(1)
addr  = 0x2D

def read16(reg):
    lo = bus.read_byte_data(addr, reg)
    hi = bus.read_byte_data(addr, reg + 1)
    return (hi << 8) | lo

while True:
    millivolt = read16(0x36) * 1.25          # Vbat  (1 mV / bit)
    milliamp  = read16(0x38)                 # Ibat  (1 mA / bit, sign shows charge/discharge)
    milliwatt = read16(0x3A)                 # Pbat  (1 mW / bit)
    print(f"{millivolt/1000:.3f} V  {milliamp} mA  {milliwatt} mW")
    time.sleep(1)
