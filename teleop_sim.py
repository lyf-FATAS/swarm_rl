import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Keyboard teleoperation for quadcopter environments."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch  # type: ignore
import numpy as np  # type: ignore

import env
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.devices import Se3Keyboard


def delta_pose_to_action(delta_pose: np.ndarray) -> np.ndarray:
    action = np.zeros(4)

    if delta_pose[4] > 0:
        action[0] = 0.1 * args_cli.sensitivity
    elif delta_pose[4] < 0:
        action[0] = -0.1 * args_cli.sensitivity

    if delta_pose[0] > 0:
        action[1] = 0.005 * args_cli.sensitivity
    elif delta_pose[0] < 0:
        action[1] = -0.005 * args_cli.sensitivity

    if delta_pose[1] > 0:
        action[2] = -0.005 * args_cli.sensitivity
    elif delta_pose[1] < 0:
        action[2] = 0.005 * args_cli.sensitivity

    if delta_pose[2] > 0:
        action[3] = 0.005 * args_cli.sensitivity
    elif delta_pose[2] < 0:
        action[3] = -0.005 * args_cli.sensitivity

    return action


def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        "FAST-Quadcopter-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("FAST-Quadcopter-Direct-v0", cfg=env_cfg)

    # create controller
    teleop_interface = Se3Keyboard()
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, _ = teleop_interface.advance()
            action = delta_pose_to_action(delta_pose)
            action = action.astype("float32")
            # convert to torch
            actions = torch.tensor(action, device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )

            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
