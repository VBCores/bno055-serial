import contextlib
import logging
import sys
import struct as st
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import serial
import numpy as np
from numpy.typing import NDArray


log = logging.getLogger(__name__)

# BOSCH BNO055 IMU Registers map and other information
# Page 0 registers
CHIP_ID = 0x00
PAGE_ID = 0x07
ACCEL_DATA = 0x08
MAG_DATA = 0x0e
GYRO_DATA = 0x14
FUSED_EULER = 0x1a
FUSED_QUAT = 0x20
LIA_DATA = 0x28
GRAVITY_DATA = 0x2e
TEMP_DATA = 0x34
CALIB_STAT = 0x35
SYS_STATUS = 0x39
SYS_ERR = 0x3a
UNIT_SEL = 0x3b
OPER_MODE = 0x3d
PWR_MODE = 0x3e
SYS_TRIGGER = 0x3f
TEMP_SOURCE = 0x440
AXIS_MAP_CONFIG = 0x41
AXIS_MAP_SIGN = 0x42

ACC_OFFSET = 0x55
MAG_OFFSET = 0x5b
GYR_OFFSET = 0x61
ACC_RADIUS = 0x68
MAG_RADIUS = 0x69

# Page 1 registers
ACC_CONFIG = 0x08
MAG_CONFIG = 0x09
GYR_CONFIG0 = 0x0a
GYR_CONFIG1 = 0x0b

#  Operation modes
OPER_MODE_CONFIG = 0x00
OPER_MODE_ACCONLY = 0x01
OPER_MODE_MAGONLY = 0x02
OPER_MODE_GYROONLY = 0x03
OPER_MODE_ACCMAG = 0x04
OPER_MODE_ACCGYRO = 0x05
OPER_MODE_MAGGYRO = 0x06
OPER_MODE_AMG = 0x07
OPER_MODE_IMU = 0x08
OPER_MODE_COMPASS = 0x09
OPER_MODE_M4G = 0x0a
OPER_MODE_NDOF_FMC_OFF = 0x0b
OPER_MODE_NDOF = 0x0C

#  Power modes
PWR_MODE_NORMAL = 0x00
PWR_MODE_LOW = 0x01
PWR_MODE_SUSPEND  = 0x02

# Communication constants
BNO055_ID = 0xa0
START_BYTE_WR = 0xaa
START_BYTE_RESP = 0xbb
START_BYTE_ERR = 0xee
READ = 0x01
WRITE = 0x00

# Read data from IMU
def read_from_dev(ser, reg_addr, length):
    buf_out = bytearray()
    buf_out.append(START_BYTE_WR)
    buf_out.append(READ)
    buf_out.append(reg_addr)
    buf_out.append(length)

    try:
        ser.write(buf_out)
    except:
        return 0

    buf_in = bytearray()
    for _ in range(5):
        with contextlib.suppress(Exception):
            buf_in.extend(ser.read((2+length)-buf_in.__len__()))


        # Check if response is correct
        if buf_in.__len__() == 0:
            continue

        if buf_in[0] == START_BYTE_ERR:
            #log.warning("Something bad when reading")
            return 0

        if buf_in[0] != START_BYTE_RESP:
            log.warning("Incorrect Bosh IMU device response.")
            return 0

        if buf_in.__len__() == 2 + length:
            buf_in.pop(0)
            buf_in.pop(0)
            break

    return buf_in


# Write data to IMU
def write_to_dev(ser, reg_addr, length, data) -> bool:
    buf_out = bytearray()
    buf_out.append(START_BYTE_WR)
    buf_out.append(WRITE)
    buf_out.append(reg_addr)
    buf_out.append(length)
    buf_out.append(data)

    try:
        ser.write(buf_out)
        buf_in = bytearray(ser.read(2))
        # print("Writing, wr: ", binascii.hexlify(buf_out), "  re: ", binascii.hexlify(buf_in))
    except:
        return False

    if (buf_in.__len__() != 2) or (buf_in[1] != 0x01):
        log.warning("Incorrect Bosh IMU device response.")
        return False
    return True


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    @cached_property
    def as_numpy(self) -> NDArray:
        return np.asarray([self.x, self.y, self.z], dtype=np.float32)


@dataclass(frozen=True)
class Quaternion(Vector3):
    w: float

    @cached_property
    def as_numpy(self) -> NDArray:
        return np.asarray([self.x, self.y, self.z, self.w], dtype=np.float32)


@dataclass(frozen=True)
class IMUData:
    orientation: Quaternion
    linear_acceleration: Vector3
    angular_velocity: Vector3
    magnetic_field: Vector3


# Factors for unit conversions
acc_fact = 1000.0
mag_fact = 16.0
gyr_fact = 900.0


def parse(byte1, byte2):
    return float(st.unpack('h', st.pack('BB', byte1, byte2))[0])


class IMUReader:
    def __init__(self, port='/dev/ttyUSB0'):
        frequency = 300
        operation_mode = OPER_MODE_NDOF
        axis_remap_config = 0x24
        axis_remap_sign = 0x00

        log.info("Opening serial port: %s...", port)
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.02)
        except serial.serialutil.SerialException:
            log.error("IMU not found at port " + port + ". Check the port in the launch file.")
            raise

        buf = read_from_dev(self.ser, CHIP_ID, 1)
        if buf == 0 or buf[0] != BNO055_ID:
            #rospy.logerr("Device ID is incorrect. Shutdown.")
            sys.exit(0)

        # IMU Configuration
        if not(write_to_dev(self.ser, OPER_MODE, 1, OPER_MODE_CONFIG)):
            log.error("Unable to set IMU into config mode.")

        if not(write_to_dev(self.ser, PWR_MODE, 1, PWR_MODE_NORMAL)):
            log.error("Unable to set IMU normal power mode.")

        if not(write_to_dev(self.ser, PAGE_ID, 1, 0x01)):
            log.error("Unable to set IMU register page 0.")

        if not(write_to_dev(self.ser, ACC_CONFIG, 1, 0x0b)):
            log.error("Unable to set IMU accelerometr units")

        if not(write_to_dev(self.ser, PAGE_ID, 1, 0x00)):
            log.error("Unable to set IMU register page 0.")

        if not(write_to_dev(self.ser, SYS_TRIGGER, 1, 0x00)):
            log.error("Unable to start IMU.")

        if not(write_to_dev(self.ser, UNIT_SEL, 1, 0x83)):
            log.error("Unable to set IMU units.")

        if not(write_to_dev(self.ser, AXIS_MAP_CONFIG, 1, axis_remap_config)):
            log.error("Unable to remap IMU axis.")

        if not(write_to_dev(self.ser, AXIS_MAP_SIGN, 1, axis_remap_sign)):
            log.error("Unable to set IMU axis signs.")

        if not(write_to_dev(self.ser, OPER_MODE, 1, operation_mode)):
            log.error("Unable to set IMU operation mode into operation mode.")

        log.info("Bosch BNO055 IMU configuration complete.")

        self.rate = 1 / frequency

    def get_data(self) -> Optional[IMUData]:
        buf = read_from_dev(self.ser, ACCEL_DATA, 45)
        if buf is None or buf == 0:
            return None

        imu_data = IMUData(
            orientation=Quaternion(
                w=parse(buf[24], buf[25]),
                x=parse(buf[26], buf[27]),
                y=parse(buf[28], buf[29]),
                z=parse(buf[30], buf[31]),
            ),
            linear_acceleration=Vector3(
                x=parse(buf[32], buf[33]) / acc_fact,
                y=parse(buf[34], buf[35]) / acc_fact,
                z=parse(buf[36], buf[37]) / acc_fact,
            ),
            angular_velocity=Vector3(
                x=parse(buf[12], buf[13]) / gyr_fact,
                y=parse(buf[14], buf[15]) / gyr_fact,
                z=parse(buf[16], buf[17]) / gyr_fact,
            ),
            magnetic_field=Vector3(
                x=parse(buf[6], buf[7]) / mag_fact,
                y=parse(buf[8], buf[9]) / mag_fact,
                z=parse(buf[10], buf[11]) / mag_fact,
            )
        )

        return imu_data

    def close(self):
        self.ser.close()
