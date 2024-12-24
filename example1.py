from pprint import pprint
import time

from bno055 import IMUReader


reader = IMUReader('/dev/ttyUSB0')

while True:
    time.sleep(0.01)
    imu_data = reader.get_data()
    pprint(imu_data)
