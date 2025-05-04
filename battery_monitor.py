import smbus2
import time
import os

class UPSMonitor:
    def __init__(self):
        self.ADDR = 0x2d
        self.LOW_VOL = 3150  # mV
        self.bus = smbus2.SMBus(1)
        self.low_voltage_counter = 0

    def get_battery_status(self):
        try:
            # Read charging state
            data = self.bus.read_i2c_block_data(self.ADDR, 0x02, 0x01)
            charging_state = "Unknown"
            if data[0] & 0x40:
                charging_state = "Fast Charging"
            elif data[0] & 0x80:
                charging_state = "Charging"
            elif data[0] & 0x20:
                charging_state = "Discharging"
            else:
                charging_state = "Idle"

            # Read battery information
            data = self.bus.read_i2c_block_data(self.ADDR, 0x20, 0x0C)
            battery_voltage = data[0] | data[1] << 8
            current = data[2] | data[3] << 8
            if current > 0x7FFF:
                current -= 0xFFFF
            battery_percent = int(data[4] | data[5] << 8)
            remaining_capacity = data[6] | data[7] << 8

            # Read cell voltages
            data = self.bus.read_i2c_block_data(self.ADDR, 0x30, 0x08)
            cell_voltages = [
                data[0] | data[1] << 8,
                data[2] | data[3] << 8,
                data[4] | data[5] << 8,
                data[6] | data[7] << 8
            ]

            return {
                "charging_state": charging_state,
                "battery_voltage": battery_voltage,
                "battery_current": current,
                "battery_percent": battery_percent,
                "remaining_capacity": remaining_capacity,
                "cell_voltages": cell_voltages
            }

        except Exception as e:
            print(f"Error reading battery status: {e}")
            return None

    def check_low_voltage(self, cell_voltages, current):
        if any(voltage < self.LOW_VOL for voltage in cell_voltages) and current < 50:
            self.low_voltage_counter += 1
            if self.low_voltage_counter >= 30:
                print("System shutdown now")
                address = os.popen("i2cdetect -y -r 1 0x2d 0x2d | egrep '2d' | awk '{print $2}'").read()
                if address != '2d\n':
                    print("0x2d i2c address not detected, something wrong.")
                else:
                    print("If charged, the system can be powered on again")
                    os.popen("i2cset -y 1 0x2d 0x01 0x55")
                os.system("sudo poweroff")
            else:
                print(f"Voltage Low, please charge in time, otherwise it will shut down in {60-2*self.low_voltage_counter} s")
        else:
            self.low_voltage_counter = 0

def main():
    monitor = UPSMonitor()
    
    while True:
        status = monitor.get_battery_status()
        if status:
            print("\n=== Battery Status ===")
            print(f"Charging State: {status['charging_state']}")
            print(f"Battery Level: {status['battery_percent']}%")
            print(f"Battery Voltage: {status['battery_voltage']} mV")
            print(f"Current: {status['battery_current']} mA")
            print(f"Remaining Capacity: {status['remaining_capacity']} mAh")
            print("\nCell Voltages:")
            for i, voltage in enumerate(status['cell_voltages'], 1):
                print(f"Cell {i}: {voltage} mV")
            
            monitor.check_low_voltage(status['cell_voltages'], status['battery_current'])
        
        time.sleep(2)

if __name__ == "__main__":
    main() 