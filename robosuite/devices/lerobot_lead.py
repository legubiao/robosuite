from typing import Dict, List, Optional

import numpy as np
import time
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.utils import move_cursor_up, init_logging

from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.devices import Device


class LeRobotLead(Device):
    """
    Class for 'device' involving mujoco viewer and mocap bodies being dragged by user's mouse.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
    """

    def __init__(self, env, teleoperator: TeleoperatorConfig = TeleoperatorConfig,
                 active_end_effector: Optional[str] = "right"):
        super().__init__(env)

        init_logging()

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False

        self.active_end_effector = active_end_effector
        self.teleoperator = make_teleoperator_from_config(teleoperator)
        self.teleoperator.connect()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """
        print("")
        print(
            "LeRobot lead arm teleoperation device to output actions."
        )
        print("Gripper control is automatic based on teleoperator input.")
        print("")

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
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        return dict()

    def get_action_bounds(self):
        """
        Get the action space bounds for the environment.
        
        Returns:
            tuple: (low, high) where low and high are numpy arrays representing 
                   the minimum and maximum action values respectively
        """
        return self.env.action_spec

    def get_joint_position_limits(self):
        """
        Get joint position limits from the robot model.
        
        Returns:
            tuple: (joint_limits_low, joint_limits_high, joint_range) where each is a numpy array
        """
        # Get joint limits from the MuJoCo model (from XML joint range attributes)
        joint_limits = self.env.sim.model.jnt_range[self.env.robots[0]._ref_joint_indexes]
        
        joint_limits_low = joint_limits[:, 0]   # minimum values
        joint_limits_high = joint_limits[:, 1]  # maximum values
        joint_range = joint_limits_high - joint_limits_low  # range
        
        return joint_limits_low, joint_limits_high, joint_range

    def get_torque_limits(self):
        """
        Get actuator torque/control limits from the robot model.
        
        Returns:
            tuple: (torque_low, torque_high) where each is a numpy array
        """
        # Get actuator control range limits (from XML actuator ctrlrange attributes)
        torque_low = self.env.sim.model.actuator_ctrlrange[self.env.robots[0]._ref_arm_joint_actuator_indexes, 0]
        torque_high = self.env.sim.model.actuator_ctrlrange[self.env.robots[0]._ref_arm_joint_actuator_indexes, 1]
        
        return torque_low, torque_high

    def get_joint_names(self):
        """
        Get the names of the robot joints.
        
        Returns:
            list: List of joint names
        """
        joint_names = []
        for joint_id in self.env.robots[0]._ref_joint_indexes:
            joint_names.append(self.env.sim.model.joint_id2name(joint_id))
        return joint_names

    def print_joint_info(self):
        """
        Print detailed information about robot joints and their limits.
        """
        joint_names = self.get_joint_names()
        pos_low, pos_high, pos_range = self.get_joint_position_limits()
        torque_low, torque_high = self.get_torque_limits()
        
        print("\n" + "="*60)
        print("ROBOT JOINT INFORMATION")
        print("="*60)
        print(f"{'Joint Name':<20} | {'Pos Low':<10} | {'Pos High':<10} | {'Range':<10} | {'Torque Low':<12} | {'Torque High':<12}")
        print("-"*60)
        
        for i, name in enumerate(joint_names):
            print(f"{name:<20} | {pos_low[i]:<10.3f} | {pos_high[i]:<10.3f} | {pos_range[i]:<10.3f} | {torque_low[i]:<12.3f} | {torque_high[i]:<12.3f}")
        
        print("="*60)

    def _get_site_names(self) -> List[str]:
        """
        Helper function to get the names of the sites used for mocap bodies.

        TODO: unify this logic to be controller independent.

        Returns:
            List[str]: A list of site names.
        """
        if isinstance(self.env.robots[0].composite_controller, WholeBody):  # input type passed to joint_action_policy
            site_names = self.env.robots[0].composite_controller.joint_action_policy.site_names
        else:
            site_name = f"gripper0_{self.active_arm}_grip_site"
            site_names = [site_name]
        return site_names

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control
        """
        super()._reset_internal_state()
        self.active_arm_indices = [0] * len(self.all_robot_arms)
        self.active_robot = 0
        self.base_modes = [False] * len(self.all_robot_arms)

    def input2action(self) -> Dict[str, np.ndarray]:
        """
        Uses mocap body poses to determine action for robot. Obtain input_type
        (i.e. absolute actions or delta actions) and input_ref_frame (i.e. world frame, base frame or eef frame)
        from the controller itself.

        """
        # TODO: unify this logic to be independent from controller type.
        action: Dict[str, np.ndarray] = {}
        gripper_dof = self.env.robots[0].gripper[self.active_end_effector].dof
        site_names = self._get_site_names()
        
        # Get joint position limits for proper scaling
        pos_low, pos_high, pos_range = self.get_joint_position_limits()
        
        for site_name in site_names:
            target_name_prefix = "right" if "right" in site_name else "left"  # hardcoded for now

            loop_start = time.perf_counter()
            action_val = self.teleoperator.get_action()

            display_len = max(len(key) for key in action_val.keys()) if action_val else 10

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in action_val.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")

            loop_s = time.perf_counter() - loop_start
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

            move_cursor_up(len(action_val) + 5)

            # Convert motor values to action space percentage and fill into action
            # Filter out gripper motors
            filtered_motors = [(motor, value) for motor, value in action_val.items() 
                             if 'gripper' not in motor.lower()]
            
            # Convert motor values to joint position range
            motor_values = [value for motor, value in filtered_motors]
            if motor_values:
                # Motor values are percentages (-100 to +100) representing offset from zero position
                # 0% = center position, +100% = positive limit, -100% = negative limit
                motor_percentages = np.array(motor_values) / 100.0  # normalize to -1 to +1 range
                
                # Calculate center position and maximum offset from center
                center_pos = (pos_low + pos_high) / 2.0  # zero/center position
                max_offset = (pos_high - pos_low) / 2.0  # maximum offset from center
                
                # Map motor percentages to actual joint positions
                joint_positions = center_pos + motor_percentages * max_offset
                
                # Debug output (optional - uncomment to see mapping results)
                # print(f"Motor %: {motor_values} -> Joint pos: {joint_positions}")
                # print(f"Center: {center_pos}, Max offset: {max_offset}")
                
                action[target_name_prefix + "_abs"] = joint_positions
            else:
                action[target_name_prefix + "_abs"] = np.array([])

            # Use gripper value from action_val to determine grasp state
            gripper_motors = [(motor, value) for motor, value in action_val.items() 
                            if 'gripper' in motor.lower()]
            if gripper_motors:
                gripper_value = gripper_motors[0][1]  # Use the first gripper motor value
                grasp = 1 if gripper_value < 10 else -1
            else:
                # Default to open gripper if no gripper motor found
                grasp = -1
            action[f"{target_name_prefix}_gripper"] = np.array([grasp] * gripper_dof)

        # TODO: enable delta actions. Currently only abs actions.
        # now convert actions to desired frames (take from controller)
        return action

    def disconnect(self):
        self.teleoperator.disconnect()