from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from quadcopter import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterSwarmEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterSwarmEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 13.0
    decimation = 2
    num_drones = 3  # Number of drones per environment
    num_actions = 4 * num_drones  # 4 actions per drone
    num_observations = 12 * num_drones  # 12 observations per drone
    num_states = 0
    debug_vis = False

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterSwarmEnv(DirectRLEnv):
    cfg: QuadcopterSwarmEnvCfg

    def __init__(
        self, cfg: QuadcopterSwarmEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # total thrust and moment applied to the base of the quadcopters
        self.num_drones = self.cfg.num_drones
        self._actions = torch.zeros(
            self.num_envs,
            self.num_drones,
            self.cfg.num_actions // self.num_drones,
            device=self.device,
        )
        self._thrust = torch.zeros(
            self.num_envs, self.num_drones, 3, device=self.device
        )
        self._moment = torch.zeros(
            self.num_envs, self.num_drones, 3, device=self.device
        )

        # goal position
        self._desired_pos_w = torch.zeros(
            self.num_envs, self.num_drones, 3, device=self.device
        )

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }

        # get specific body indices for each drone
        self._body_id = self._robots[0].find_bodies("body")[0]
        self._robot_mass = self._robots[0].root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity, device=self.device
        ).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robots = []
        for i in range(self.num_drones):
            robot = Articulation(
                self.cfg.robot.replace(prim_path=f"/World/envs/env_.*/Robot_{i}")
            )
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.view(self.num_envs, self.num_drones, -1)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        for i in range(self.num_drones):
            self._thrust[:, i, 2] = (
                self.cfg.thrust_to_weight
                * self._robot_weight
                * (self._actions[:, i, 0] + 1.0)
                / 2.0
            )
            self._moment[:, i, :] = self.cfg.moment_scale * self._actions[:, i, 1:]

    def _apply_action(self):
        for i in range(self.num_drones):
            self._robots[i].set_external_force_and_torque(
                self._thrust[:, i, :], self._moment[:, i, :], body_ids=self._body_id
            )

    def _get_observations(self) -> dict:
        obs_list = []
        for i in range(self.num_drones):
            desired_pos_b, _ = subtract_frame_transforms(
                self._robots[i].data.root_state_w[:, :3],
                self._robots[i].data.root_state_w[:, 3:7],
                self._desired_pos_w[:, i, :],
            )
            obs = torch.cat(
                [
                    self._robots[i].data.root_lin_vel_b,
                    self._robots[i].data.root_ang_vel_b,
                    self._robots[i].data.projected_gravity_b,
                    desired_pos_b,
                ],
                dim=-1,
            )
            obs_list.append(obs)
        observations = {"policy": torch.cat(obs_list, dim=-1)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_sum = 0.0
        ang_vel_sum = 0.0
        distance_to_goal_sum = 0.0
        for i in range(self.num_drones):
            lin_vel = torch.sum(
                torch.square(self._robots[i].data.root_lin_vel_b), dim=1
            )
            ang_vel = torch.sum(
                torch.square(self._robots[i].data.root_ang_vel_b), dim=1
            )
            distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, i, :] - self._robots[i].data.root_pos_w, dim=1
            )
            distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

            lin_vel_sum += lin_vel
            ang_vel_sum += ang_vel
            distance_to_goal_sum += distance_to_goal_mapped

        rewards = {
            "lin_vel": lin_vel_sum * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sum * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_sum
            * self.cfg.distance_to_goal_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_drones):
            died = died | torch.logical_or(
                self._robots[i].data.root_pos_w[:, 2] < -0.1,
                self._robots[i].data.root_pos_w[:, 2] > 5.2,
            )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # logging
        final_distance_to_goal_sum = 0.0
        for i in range(self.num_drones):
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[env_ids, i, :]
                - self._robots[i].data.root_pos_w[env_ids],
                dim=1,
            ).mean()
            final_distance_to_goal_sum += final_distance_to_goal

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal_sum.item()
        self.extras["log"].update(extras)

        for i in range(self.num_drones):
            self._robots[i].reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        # sample new commands for each drone
        for i in range(self.num_drones):
            self._desired_pos_w[env_ids, i, :2] = torch.zeros_like(
                self._desired_pos_w[env_ids, i, :2]
            ).uniform_(-2.0, 2.0)
            self._desired_pos_w[env_ids, i, :2] += self._terrain.env_origins[
                env_ids, :2
            ]
            self._desired_pos_w[env_ids, i, 2] = torch.zeros_like(
                self._desired_pos_w[env_ids, i, 2]
            ).uniform_(0.5, 1.5)
            # Reset robot state
            joint_pos = self._robots[i].data.default_joint_pos[env_ids]
            joint_vel = self._robots[i].data.default_joint_vel[env_ids]
            default_root_state = self._robots[i].data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self._robots[i].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robots[i].write_root_velocity_to_sim(
                default_root_state[:, 7:], env_ids
            )
            self._robots[i].write_joint_state_to_sim(
                joint_pos, joint_vel, None, env_ids
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


import gymnasium as gym

gym.register(
    id="FAST-Quadcopter-Swarm-Direct-v0",
    entry_point=QuadcopterSwarmEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterSwarmEnvCfg,
    },
)
