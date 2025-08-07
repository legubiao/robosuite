from .device import Device
from .keyboard import Keyboard

try:
    from .spacemouse import SpaceMouse
    from .dualsense import DualSense
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse or DualSense.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )

try:
    from .ros2_joy import ROS2Joy
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load ROS2Joy device. Install rclpy to use ROS2 joy controller.\n
           Install with: pip install rclpy"""
    )