from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class PickPlaceTable(ManipulationEnv):
    """
    Multi-object pick-and-place on a flat table.

    Spawns multiple graspable cubes on the table and visual circular target markers on the table surface.
    The goal is to pick each cube and place it at its corresponding target marker.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        goal_radius=0.06,
        num_objects=3,
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
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = np.array(table_full_size)
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # observables
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # goal marker radius (visual target on table)
        self.goal_radius = float(goal_radius)
        
        # number of objects to manipulate
        self.num_objects = int(num_objects)

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
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
            - 1.0 per object if it is placed at its corresponding target marker.

        If reward shaping is enabled, staged rewards are added:
            - Reaching, Grasping, Lifting, Hovering-above-goal
        """
        reward = 0.0

        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_at_goals)

        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)

        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards for shaping: reach, grasp, lift, hover-above-goal
        """
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already at their goals
        active_objs = []
        for i, obj in enumerate(self.objects):
            if self.objects_at_goals[i]:
                continue
            active_objs.append(obj)

        # reaching: distance from gripper to closest object
        r_reach = 0.0
        if active_objs:
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper, target=active_obj.root_body, target_type="body", return_distance=True
                )
                for active_obj in active_objs
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping: nonzero if grasp detected
        r_grasp = 0.0
        if active_objs:
            r_grasp = float(self._check_grasp(
                gripper=self.robots[0].gripper, 
                object_geoms=[g for active_obj in active_objs for g in active_obj.contact_geoms]
            )) * grasp_mult

        # lifting: only if grasping, proportion to height up to a small target above table
        r_lift = 0.0
        if active_objs and r_grasp > 0.0:
            object_heights = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][:, 2]
            table_height = self.table_offset[2]
            z_target = table_height + 0.20
            z_dists = np.maximum(z_target - object_heights, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover: closeness in XY of cube to goal center, with higher reward if lifted
        r_hover = 0.0
        if active_objs:
            target_goal_ids = [i for i, obj in enumerate(self.objects) if obj in active_objs]
            goal_xy = self.sim.data.body_xpos[[self.goal_body_ids[i] for i in target_goal_ids]][:, :2]
            obj_xy = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][:, :2]
            xy_dists = np.linalg.norm(goal_xy - obj_xy, axis=1)
            base = r_lift if r_lift > 0.0 else r_reach
            r_hover = base + (1 - np.tanh(10.0 * min(xy_dists))) * (hover_mult - base)

        return r_reach, r_grasp, r_lift, r_hover

    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        for i in range(self.num_objects):
            obj = BoxObject(
                name=f"cube_{i}",
                size_min=[0.020, 0.020, 0.020],
                size_max=[0.025, 0.025, 0.025],
                rgba=[0.9, 0.1, 0.1, 1.0],
                material=None,
            )
            self.objects.append(obj)

    def _construct_goal_markers(self):
        """
        Function that can be overriden by subclasses to load different goal markers.
        """
        self.goal_markers = []
        colors = [
            [0.1, 0.6, 1.0, 0.6],  # blue
            [0.1, 0.8, 0.1, 0.6],  # green  
            [0.8, 0.1, 0.8, 0.6],  # magenta
            [1.0, 0.6, 0.1, 0.6],  # orange
            [0.6, 0.1, 0.6, 0.6],  # purple
        ]
        
        for i in range(self.num_objects):
            color = colors[i % len(colors)]
            goal_marker = CylinderObject(
                name=f"VisualGoal_{i}",
                size=[self.goal_radius, 0.001],  # radius, half-length
                rgba=color,
                joints=None,
                obj_type="visual",
            )
            self.goal_markers.append(goal_marker)

    def _load_model(self):
        """Loads XML model for this environment."""
        super()._load_model()

        # Adjust base pose accordingly (same pattern as Microwave / Lift)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Arena with single table
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Construct objects and goal markers
        self._construct_objects()
        self._construct_goal_markers()

        # Task composition
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.goal_markers,
        )

        # Placement initializer (objects + visual goals)
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
        else:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectAndGoalSampler")

            # Sample objects in the left half of the table
            table_x_half = self.table_full_size[0] / 2 - 0.05
            table_y_half = self.table_full_size[1] / 2 - 0.05
            
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="ObjectsSampler",
                    mujoco_objects=self.objects,
                    x_range=[-table_x_half, -0.05],  # left side of table
                    y_range=[-table_y_half, table_y_half],
                    rotation=(-np.pi, np.pi),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )
            )

            # Sample goal markers in the right half of the table
            for i, goal_marker in enumerate(self.goal_markers):
                # Distribute goals across the right side of the table
                x_pos = 0.05 + (i * 0.15)  # spread goals horizontally
                if x_pos > table_x_half:
                    x_pos = table_x_half - 0.05
                
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"GoalSampler_{i}",
                        mujoco_objects=goal_marker,
                        x_range=[x_pos, x_pos],  # fixed x position
                        y_range=[-table_y_half + 0.05, table_y_half - 0.05],
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.table_offset,
                        z_offset=0.0,
                    )
                )

    def _setup_references(self):
        """Sets up references to important components."""
        super()._setup_references()

        # object and goal body ids
        self.obj_body_id = {}
        self.obj_geom_id = {}
        self.goal_body_ids = []

        # object-specific ids
        for obj in self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # goal marker body ids
        for goal_marker in self.goal_markers:
            self.goal_body_ids.append(self.sim.model.body_name2id(goal_marker.root_body))

        # keep track of which objects are at their goals
        self.objects_at_goals = np.zeros(len(self.objects))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            # object-related observables
            sensors = []
            names = []
            
            for obj in self.objects:
                @sensor(modality=modality)
                def obj_pos(obs_cache, obj_name=obj.name):
                    return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

                @sensor(modality=modality)
                def obj_quat(obs_cache, obj_name=obj.name):
                    return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

                sensors.extend([obj_pos, obj_quat])
                names.extend([f"{obj.name}_pos", f"{obj.name}_quat"])

            # goal marker observables
            for i, goal_marker in enumerate(self.goal_markers):
                @sensor(modality=modality)
                def goal_pos(obs_cache, goal_idx=i):
                    return np.array(self.sim.data.body_xpos[self.goal_body_ids[goal_idx]])

                sensors.append(goal_pos)
                names.append(f"goal_{i}_pos")

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to object position sensors; one for each arm and object
            for obj in self.objects:
                sensors += [
                    self._get_obj_eef_sensor(full_pf, f"{obj.name}_pos", f"{arm_pf}gripper_to_{obj.name}_pos", modality)
                    for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                ]
                names += [f"{arm_pf}gripper_to_{obj.name}_pos" for arm_pf in arm_prefixes]

            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        """Resets simulation internal configurations."""
        super()._reset_internal()

        if not self.deterministic_reset:
            # Sample from the placement initializer for objects and goals
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if "visual" in obj.name.lower():
                    # Visual object: set body pose directly
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = np.array(obj_pos)
                    self.sim.model.body_quat[body_id] = np.array(obj_quat)
                else:
                    # Physical object: set free joint qpos
                    self.sim.data.set_joint_qpos(
                        obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                    )

    def _check_success(self):
        """
        Check if all objects have been placed at their corresponding goal markers on the table.
        """
        # remember objects that are at their goals
        for i, obj in enumerate(self.objects):
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj.name]]
            goal_pos = self.sim.data.body_xpos[self.goal_body_ids[i]]

            # XY distance within goal radius
            xy_close = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < self.goal_radius * 0.6

            # Close to tabletop height (object center should be at table height + half-height)
            half_z = float(obj.size[2])
            table_height = float(self.table_offset[2])
            z_target = table_height + half_z
            z_close = abs(float(obj_pos[2]) - z_target) < 0.02

            self.objects_at_goals[i] = int(xy_close and z_close)

        # returns True if all objects are at their goals
        return np.sum(self.objects_at_goals) == len(self.objects)

    def visualize(self, vis_settings):
        """
        Visualize gripper-to-closest-object distance using site coloring.
        """
        super().visualize(vis_settings=vis_settings)

        if vis_settings["grippers"]:
            # find closest object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper, target=obj.root_body, target_type="body", return_distance=True
                )
                for obj in self.objects
            ]
            closest_obj_id = np.argmin(dists)
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.objects[closest_obj_id]
            )


