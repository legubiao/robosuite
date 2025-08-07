"""
Driver class for ROS2 Joy controller.
"""

import numpy as np
import threading
import time
from typing import Dict, Optional

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available. Install rclpy to use ROS2Joy device.")

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class ROS2JoyNode(Node):
    """
    ROS2 node for subscribing to /joy topic
    """

    def __init__(self):
        super().__init__('robosuite_joy_node')
        self.joy_subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        self.latest_joy_msg = None
        self.lock = threading.Lock()

    def joy_callback(self, msg):
        """Callback for joy messages"""
        with self.lock:
            self.latest_joy_msg = msg

    def get_latest_joy_msg(self):
        """Get the latest joy message thread-safely"""
        with self.lock:
            return self.latest_joy_msg


class ROS2Joy(Device):
    """
    A driver class for ROS2 Joy controller.
    
    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
        joy_topic (str): ROS2 topic name for joy messages (default: '/joy')
        mapping_config (dict): Configuration for mapping joystick axes/buttons to actions
    """

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0,
                 joy_topic='/joy', mapping_config=None):
        super().__init__(env)

        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 is not available. Please install rclpy.")

        # ROS2 setup
        self.joy_topic = joy_topic
        self.ros2_node = None
        self.ros2_thread = None
        self.ros2_running = False

        # Default mapping configuration
        self.mapping_config = mapping_config or {
            # Axes mapping (for your specific controller)
            'left_stick_x': 0,  # Left stick horizontal
            'left_stick_y': 1,  # Left stick vertical
            'right_stick_x': 3,  # Right stick horizontal (adjusted)
            'right_stick_y': 4,  # Right stick vertical (adjusted)
            'left_trigger': 2,  # Left trigger (adjusted)
            'right_trigger': 5,  # Right trigger (adjusted)
            'dpad_x': 6,  # D-pad horizontal
            'dpad_y': 7,  # D-pad vertical

            # Buttons mapping
            'a': 0,  # A button (grasp toggle)
            'b': 1,  # B button (reset)
            'x': 2,  # X button (switch arm)
            'y': 3,  # Y button (switch robot)
            'lb': 4,  # Left bumper (base mode toggle)
            'rb': 5,  # Right bumper (torso mode toggle)
            'back': 6,  # Back button
            'start': 7,  # Start button
            'left_stick_press': 8,  # Left stick press
            'right_stick_press': 9,  # Right stick press
        }

        # Button state tracking
        self.button_states = {}
        self.prev_button_states = {}

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # Start ROS2 node in a separate thread
        self._start_ros2_node()

    def _start_ros2_node(self):
        """Start ROS2 node in a separate thread"""

        def ros2_spin():
            rclpy.init()
            self.ros2_node = ROS2JoyNode()
            self.ros2_running = True

            while self.ros2_running:
                rclpy.spin_once(self.ros2_node, timeout_sec=0.01)
                time.sleep(0.01)

            if self.ros2_node:
                self.ros2_node.destroy_node()
            rclpy.shutdown()

        self.ros2_thread = threading.Thread(target=ros2_spin, daemon=True)
        self.ros2_thread.start()

        # Wait a bit for ROS2 to initialize
        time.sleep(1.0)

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(control, info):
            control += " " * (30 - len(control))
            print("{}\t{}".format(control, info))

        print("")
        print_command("ROS2 Joy Control", "Command")
        print_command("Left Stick", "move horizontally in x-y plane")
        print_command("Right Stick X", "rotate (yaw)")
        print_command("Right Stick Y", "move vertically (z-axis)")
        print_command("D-pad X", "rotate (roll)")
        print_command("D-pad Y", "rotate (pitch)")
        print_command("A Button", "switch active arm (if multi-armed robot)")
        print_command("B Button", "reset simulation")
        print_command("X Button", "toggle gripper (open/close)")
        print_command("Y Button", "switch active robot (if multi-robot environment)")
        print_command("LB Button", "toggle arm/base mode (if applicable)")
        print_command("RB Button", "toggle torso mode (if applicable)")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)

        # Initialize button states
        for button_name in self.mapping_config.values():
            if isinstance(button_name, int) and button_name < 20:  # Assuming max 20 buttons
                self.button_states[button_name] = False
                self.prev_button_states[button_name] = False

        # Initialize D-pad state
        self.dpad_x = 0.0
        self.dpad_y = 0.0

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the ROS2 joy controller.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        if not self.ros2_node:
            # Return default state if ROS2 is not available
            return dict(
                dpos=np.zeros(3),
                rotation=self.rotation,
                raw_drotation=np.zeros(3),
                grasp=int(self.grasp),
                reset=self._reset_state,
                base_mode=int(self.base_mode),
                torso_mode=int(self.torso_mode),
            )

        # Get latest joy message
        joy_msg = self.ros2_node.get_latest_joy_msg()

        if joy_msg is not None:
            # Process axes
            self._process_axes(joy_msg.axes)

            # Process buttons
            self._process_buttons(joy_msg.buttons)

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
                self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
            torso_mode=int(self.torso_mode),
        )

    def _process_axes(self, axes):
        """Process joystick axes"""
        if len(axes) < 6:
            return

        # Position control (left stick for x-y, triggers for z)
        left_stick_x = axes[self.mapping_config['left_stick_x']]
        left_stick_y = axes[self.mapping_config['left_stick_y']]

        # Apply deadzone
        deadzone = 0.1
        if abs(left_stick_x) < deadzone:
            left_stick_x = 0.0
        if abs(left_stick_y) < deadzone:
            left_stick_y = 0.0

        # Update position
        self.pos[1] -= left_stick_x * self._pos_step * self.pos_sensitivity
        self.pos[0] -= left_stick_y * self._pos_step * self.pos_sensitivity

        # Note: Z-axis control is now handled by right stick Y-axis
        # Triggers are reserved for future use

        # Right stick control (Yaw and Z-axis)
        right_stick_x = axes[self.mapping_config['right_stick_x']]
        right_stick_y = axes[self.mapping_config['right_stick_y']]

        # Apply deadzone
        if abs(right_stick_x) < deadzone:
            right_stick_x = 0.0
        if abs(right_stick_y) < deadzone:
            right_stick_y = 0.0

        # Yaw control with right stick X-axis
        if abs(right_stick_x) > 0:
            drot = rotation_matrix(angle=right_stick_x * 0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[2] += right_stick_x * 0.1 * self.rot_sensitivity

        # Z-axis control with right stick Y-axis
        if abs(right_stick_y) > 0:
            self.pos[2] += right_stick_y * self._pos_step * self.pos_sensitivity

        # Process D-pad (directional pad)
        if len(axes) >= 8:
            dpad_x = axes[self.mapping_config['dpad_x']]
            dpad_y = axes[self.mapping_config['dpad_y']]

            # Apply deadzone to D-pad
            dpad_deadzone = 0.5
            if abs(dpad_x) < dpad_deadzone:
                dpad_x = 0.0
            if abs(dpad_y) < dpad_deadzone:
                dpad_y = 0.0

            # Roll control with D-pad X-axis
            if abs(dpad_x) > 0:
                drot = rotation_matrix(angle=dpad_x * 0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)
                self.raw_drotation[1] += dpad_x * 0.1 * self.rot_sensitivity

            # Pitch control with D-pad Y-axis (inverted)
            if abs(dpad_y) > 0:
                drot = rotation_matrix(angle=-dpad_y * 0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)
                self.raw_drotation[0] += -dpad_y * 0.1 * self.rot_sensitivity

    def _process_buttons(self, buttons):
        """Process joystick buttons"""
        if len(buttons) < 10:
            return

        # Define button mappings separately from axis mappings
        button_mappings = {
            'a': 0,  # A button (switch arm)
            'b': 1,  # B button (reset)
            'x': 2,  # X button (grasp toggle)
            'y': 3,  # Y button (switch robot)
            'lb': 4,  # Left bumper (base mode toggle)
            'rb': 5,  # Right bumper (torso mode toggle)
            'back': 6,  # Back button
            'start': 7,  # Start button
            'left_stick_press': 8,  # Left stick press
            'right_stick_press': 9,  # Right stick press
        }

        # Update button states
        for button_name, button_idx in button_mappings.items():
            if button_idx < len(buttons):
                self.prev_button_states[button_idx] = self.button_states.get(button_idx, False)
                self.button_states[button_idx] = buttons[button_idx] > 0

                # Check for button press (rising edge)
                if self.button_states[button_idx] and not self.prev_button_states[button_idx]:
                    print(f"Button pressed: {button_name} (index {button_idx})")
                    self._handle_button_press(button_name)

    def _handle_button_press(self, button_name):
        """Handle button press events"""
        if button_name == 'a':  # Switch arm
            self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
            print(f"Switched to arm {self.active_arm_index}")

        elif button_name == 'b':  # Reset
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()
            print("Reset triggered!")

        elif button_name == 'x':  # Grasp toggle
            self.grasp_states[self.active_robot][self.active_arm_index] = not self.grasp_states[self.active_robot][
                self.active_arm_index]
            print(f"Gripper toggled: {self.grasp_states[self.active_robot][self.active_arm_index]}")

        elif button_name == 'y':  # Switch robot
            self.active_robot = (self.active_robot + 1) % self.num_robots
            print(f"Switched to robot {self.active_robot}")

        elif button_name == 'lb':  # Base mode toggle
            self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]
            mode = "base" if self.base_modes[self.active_robot] else "arm"
            print(f"Switched to {mode} mode")

        elif button_name == 'rb':  # Torso mode toggle
            self.torso_modes[self.active_robot] = not self.torso_modes[self.active_robot]
            mode = "enabled" if self.torso_modes[self.active_robot] else "disabled"
            print(f"Torso mode {mode}")

    def _postprocess_device_outputs(self, dpos, drotation):
        """Post-process device outputs with scaling"""
        drotation = drotation * 1.5
        dpos = dpos * 10  # Further reduced from 25 to 10 for much lower position sensitivity

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation

    def __del__(self):
        """Cleanup when device is destroyed"""
        if hasattr(self, 'ros2_running'):
            self.ros2_running = False
        if hasattr(self, 'ros2_thread') and self.ros2_thread and self.ros2_thread.is_alive():
            self.ros2_thread.join(timeout=1.0)
