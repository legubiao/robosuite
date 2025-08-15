import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BreadObject,
    BreadVisualObject,
    CanObject,
    CanVisualObject,
    CerealObject,
    CerealVisualObject,
    MilkObject,
    MilkVisualObject,
    Shelf8Object,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class PickPlaceTable(ManipulationEnv):
    """
    This class corresponds to the pick place task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        bin1_pos (3-tuple): Absolute cartesian coordinates of the bin initially holding the objects

        bin2_pos (3-tuple): Absolute cartesian coordinates of the goal bin

        z_offset (float): amount of z offset for initializing objects in bin

        z_rotation (float, tuple, or None): if provided, controls the range of z-rotation initialization
            for the objects

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        single_object_mode (int): specifies which version of the task to do. Note that
            the observations change accordingly.

            :`0`: corresponds to the full task with all types of objects.

            :`1`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is randomized on every reset.

            :`2`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.

        object_type (string): if provided, should be one of "milk", "bread", "cereal",
            or "can". Determines which type of object will be spawned on every
            environment reset. Only used if @single_object_mode is 2.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid object type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(1.0, 1.5, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        z_offset=0.0,
        z_rotation=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=0,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.object_id_to_sensors = {}  # Maps object id to sensor names for that object
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        if object_type is not None:
            assert object_type in self.object_to_id.keys(), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[object_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

        # settings for object placement
        self.z_offset = z_offset  # z offset for initializing items on table
        self.z_rotation = z_rotation  # z rotation for initializing items on table

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

          - a discrete reward of 1.0 per object if it is placed in its correct table position

        Un-normalized components if using reward shaping, where the maximum is returned if not solved:

          - Reaching: in [0, 0.1], proportional to the distance between the gripper and the closest object
          - Grasping: in {0, 0.35}, nonzero if the gripper is grasping an object
          - Lifting: in {0, [0.35, 0.5]}, nonzero only if object is grasped; proportional to lifting height
          - Hovering: in {0, [0.5, 0.7]}, nonzero only if object is lifted; proportional to distance from object to table target area

        Note that a successfully completed task (object in correct table position) will return 1.0 per object irregardless of whether the
        environment is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 4.0 (or 1.0 if only a single object is
        being used) as well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_on_table)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 4.0
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct table positions
        active_objs = []
        for obj in self.objects:
            # Skip shelf8 object since it's static
            if obj.name == "shelf8":
                continue
            # Use the mapping to get the correct index in objects_on_table
            if self.objects_on_table[self.obj_to_table_index[obj.name]]:
                continue
            active_objs.append(obj)

        # reaching reward governed by distance to closest object
        r_reach = 0.0
        if active_objs:
            # get reaching reward via minimum distance to a target object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_obj.root_body,
                    target_type="body",
                    return_distance=True,
                )
                for active_obj in active_objs
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[g for active_obj in active_objs for g in active_obj.contact_geoms],
                )
            )
            * grasp_mult
        )

        # lifting reward for picking up an object
        r_lift = 0.0
        if active_objs and r_grasp > 0.0:
            z_target = self.table_top_pos[2] + self.table_size[2] / 2.0 + 0.25
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, 2
            ]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover reward for getting object above table target area
        r_hover = 0.0
        if active_objs:
            target_table_ids = [self.object_to_id[active_obj.name.lower()] for active_obj in active_objs]
            # segment objects into left of the table target areas and above the table target areas
            object_xy_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, :2
            ]
            y_check = (
                np.abs(object_xy_locs[:, 1] - self.target_table_placements[target_table_ids, 1]) < self.table_size[1] / 4.0
            )
            x_check = (
                np.abs(object_xy_locs[:, 0] - self.target_table_placements[target_table_ids, 0]) < self.table_size[0] / 4.0
            )
            objects_above_targets = np.logical_and(x_check, y_check)
            objects_not_above_targets = np.logical_not(objects_above_targets)
            dists = np.linalg.norm(self.target_table_placements[target_table_ids, :2] - object_xy_locs, axis=1)
            # objects to the left get r_lift added to hover reward,
            # those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(active_objs))
            r_hover_all[objects_above_targets] = lift_mult + (1 - np.tanh(10.0 * dists[objects_above_targets])) * (
                hover_mult - lift_mult
            )
            r_hover_all[objects_not_above_targets] = r_lift + (1 - np.tanh(10.0 * dists[objects_not_above_targets])) * (
                hover_mult - lift_mult
            )
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_on_table(self, obj_pos, obj_id):
        """
        Check if object is not in the correct table target area.
        
        Args:
            obj_pos (np.array): Object position
            obj_id (int): Object ID
            
        Returns:
            bool: True if object is not in correct target area
        """
        target_pos = self.target_table_placements[obj_id]
        table_half_size = self.table_size / 2.0
        
        # Check if object is within the target area bounds
        x_check = abs(obj_pos[0] - target_pos[0]) < table_half_size[0] / 2.0
        y_check = abs(obj_pos[1] - target_pos[1]) < table_half_size[1] / 2.0
        z_check = (target_pos[2] - 0.05 < obj_pos[2] < target_pos[2] + 0.1)
        
        return not (x_check and y_check and z_check)

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere on table surface
        table_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        table_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        # Place shelf8 on the left side of the table (fixed position but movable object)
        # Position shelf8 so its surface is level with the table surface
        shelf8_x = table_x_half - 0.4  # Slightly to the left of table edge
        shelf8_y = -table_y_half + 0.3  # Center of table width
        # Adjust z position so shelf8 surface aligns with table surface
        # shelf8 has height of 0.6 (from bottom_site to top_site), so center it at table height
        shelf8_z = self.table_top_pos[2]  # Center shelf8 at table height
        
        # Add shelf8 placement sampler (fixed position)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="Shelf8Sampler",
                mujoco_objects=[self.shelf8],
                x_range=[shelf8_x, shelf8_x],
                y_range=[shelf8_y, shelf8_y],
                rotation=0.0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_top_pos,
                z_offset=shelf8_z - self.table_top_pos[2],
                rng=self.rng,
            )
        )

        # other objects should be sampled in the bounds of the table (with some tolerance)
        other_objects = [obj for obj in self.objects if obj.name != "shelf8"]
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=other_objects,
                x_range=[-table_x_half, table_x_half],  # Full length of table (front-back)
                y_range=[0, table_y_half],  # Only right side of table (y > 0)
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_top_pos,
                z_offset=self.z_offset,
                rng=self.rng,
            )
        )

        # each visual object should just be at the center of each target table area
        index = 0
        for vis_obj in self.visual_objects:
            # get center of target table area
            target_pos = self.target_table_placements[index]
            
            # placement is relative to table center, so compute difference and send to placement initializer
            rel_center = target_pos[:2] - self.table_top_pos[:2]

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{vis_obj.name}ObjectSampler",
                    mujoco_objects=vis_obj,
                    x_range=[rel_center[0], rel_center[0]],
                    y_range=[rel_center[1], rel_center[1]],
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_top_pos,
                    z_offset=target_pos[2] - self.table_top_pos[2],
                    rng=self.rng,
                )
            )
            index += 1

    def _construct_visual_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
            (MilkVisualObject, BreadVisualObject, CerealVisualObject, CanVisualObject),
            self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        
        # Add shelf8 object on the left side of the table
        self.shelf8 = Shelf8Object(name="shelf8")
        self.objects.append(self.shelf8)
        
        # Add the original objects
        for obj_cls, obj_name in zip(
            (MilkObject, BreadObject, CerealObject, CanObject),
            self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size, 
            table_friction=self.table_friction,
            table_offset=self.table_offset
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera to better observe shelf8 and pick and place task
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0, 0, 2.0],  # Higher position than original (was 1.35), centered view
            quat=[0.653, 0.271, 0.271, 0.653]  # Keep original orientation
        )

        # store some arena attributes
        self.table_size = mujoco_arena.table_full_size
        self.table_top_pos = mujoco_arena.table_top_abs

        # Create target table placements for objects
        self.target_table_placements = np.zeros((4, 3))  # 4 objects: milk, bread, cereal, can
        table_half_size = self.table_size / 2.0
        
        for i in range(4):
            # Define target placement areas on the right side of the table
            # All objects now go on the right side (y > 0)
            target_x = self.table_top_pos[0]  # Center of table length (front-back)
            
            # Alternate between top and bottom for each side
            if i % 2 == 0:  # Even indices (0, 2) - top right
                target_y = self.table_top_pos[1] + table_half_size[1] / 2.0
            else:  # Odd indices (1, 3) - bottom right
                target_y = self.table_top_pos[1] + table_half_size[1] / 4.0
            
            # Set target position on table surface
            # table_top_pos[2] is the table top surface, but we want the center of the table
            # So we need to subtract half the table height
            target_z = self.table_top_pos[2] - self.table_size[2] / 2.0 + self.z_offset
            self.target_table_placements[i, :] = [target_x, target_y, target_z]

        # make objects
        self._construct_visual_objects()
        self._construct_objects()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects,
        )

        # Generate placement initializer
        self._get_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in self.visual_objects + self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are placed on the table
        # Include shelf8 in tracking since it's now movable, but exclude it from reward calculations
        self.objects_on_table = np.zeros(len(self.objects))
        
        # Create a mapping from object names to their index in objects_on_table
        self.obj_to_table_index = {}
        for i, obj in enumerate(self.objects):
            self.obj_to_table_index[obj.name] = i

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # Reset obj sensor mappings
            self.object_id_to_sensors = {}

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [
                self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)
            enableds = [True] * len(sensors)

            for i, obj in enumerate(self.objects):
                # Create object sensors
                # For shelf8, always enable sensors but mark as not participating in rewards
                if obj.name == "shelf8":
                    using_obj = True  # Always enable sensors for shelf8
                else:
                    # For movable objects, use single_object_mode logic
                    using_obj = self.single_object_mode == 0 or self.object_id == i
                
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                enableds += [using_obj] * len(obj_sensor_names)
                actives += [using_obj] * len(obj_sensor_names)
                
                # Track all objects in object_id_to_sensors for consistency
                self.object_id_to_sensors[i] = obj_sensor_names

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def obj_id(obs_cache):
                    return self.object_id

                sensors.append(obj_id)
                names.append("obj_id")
                enableds.append(True)
                actives.append(True)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        sensors = [
            self._get_rel_obj_eef_sensor(arm_pf, obj_name, f"{obj_name}_to_{full_pf}eef_pos", full_pf, modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        sensors += [
            self._get_obj_eef_rel_quat_sensor(full_pf, obj_name, f"{obj_name}_to_{full_pf}eef_quat", modality)
            for full_pf in full_prefixes
        ]
        names = [fn.__name__ for fn in sensors]
        sensors += [obj_pos, obj_quat]
        names += [f"{obj_name}_pos", f"{obj_name}_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    # Check if object has joints (movable objects) or not (static objects like shelf8)
                    if hasattr(obj, 'joints') and obj.joints:
                        # Set the collision object joints for movable objects
                        self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                    else:
                        # For static objects without joints, set body position directly
                        self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                        self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat

        # Set the bins to the desired position
        # Note: This is no longer needed since we're using TableArena instead of BinsArena
        # The table position is set when the arena is created

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objects}
        # Remove shelf8 from objects that can be moved/cleared since it has fixed position
        if "shelf8" in obj_names:
            obj_names.remove("shelf8")
            
        if self.single_object_mode == 1:
            self.obj_to_use = self.rng.choice(list(obj_names))
            for obj_type, i in self.object_to_id.items():
                if obj_type.lower() in self.obj_to_use.lower():
                    self.object_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.object_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active object, else False
                    self._observables[name].set_enabled(i == self.object_id)
                    self._observables[name].set_active(i == self.object_id)

    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if all objects are placed correctly
        """
        # remember objects that are in the correct bins
        # Exclude shelf8 from success checking since it's not part of the task
        movable_objects = [obj for obj in self.objects if obj.name != "shelf8"]
        
        for i, obj in enumerate(movable_objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = min(
                [
                    np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos)
                    for arm in self.robots[0].arms
                ]
            )
            r_reach = 1 - np.tanh(10.0 * dist)
            # Use the mapping to get the correct index in objects_on_table
            table_index = self.obj_to_table_index[obj_str]
            # Only check objects that have target placements (exclude shelf8)
            if obj_str.lower() in self.object_to_id:
                obj_id = self.object_to_id[obj_str.lower()]
                self.objects_on_table[table_index] = int((not self.not_on_table(obj_pos, obj_id)) and r_reach < 0.6)

        # returns True if a single object is in the correct table position
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_on_table) > 0

        # returns True if all objects are in correct table positions
        return np.sum(self.objects_on_table) == len(movable_objects)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # if the robot has multiple arms color each arm independently based on its closest object
            for arm in self.robots[0].arms:
                # find closest object (exclude shelf8 since it's not part of the task)
                movable_objects = [obj for obj in self.objects if obj.name != "shelf8"]
                dists = [
                    self._gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=obj.root_body,
                        target_type="body",
                        return_distance=True,
                    )
                    for obj in movable_objects
                ]
                if dists:  # Only proceed if there are movable objects
                    closest_obj_id = np.argmin(dists)
                    # Visualize the distance to this target
                    self._visualize_gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=movable_objects[closest_obj_id].root_body,
                        target_type="body",
                    )


class PickPlaceSingle(PickPlaceTable):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class PickPlaceMilk(PickPlaceTable):
    """
    Easier version of task - place one milk into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="milk", **kwargs)


class PickPlaceBread(PickPlaceTable):
    """
    Easier version of task - place one bread into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="bread", **kwargs)


class PickPlaceCereal(PickPlaceTable):
    """
    Easier version of task - place one cereal into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="cereal", **kwargs)


class PickPlaceCan(PickPlaceTable):
    """
    Easier version of task - place one can into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="can", **kwargs)
