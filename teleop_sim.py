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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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


import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings

import env, camera_env, swarm_env
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.devices import Se3Keyboard


def delta_pose_to_action(delta_pose: np.ndarray) -> np.ndarray:
    action = np.zeros(4)

    if delta_pose[4] > 0:
        action[0] = 0.13 * args_cli.sensitivity
    elif delta_pose[4] < 0:
        action[0] = -0.13 * args_cli.sensitivity

    if delta_pose[0] > 0:
        action[2] = 0.005 * args_cli.sensitivity
    elif delta_pose[0] < 0:
        action[2] = -0.005 * args_cli.sensitivity

    if delta_pose[1] > 0:
        action[1] = 0.005 * args_cli.sensitivity
    elif delta_pose[1] < 0:
        action[1] = -0.005 * args_cli.sensitivity

    if delta_pose[2] > 0:
        action[3] = 0.005 * args_cli.sensitivity
    elif delta_pose[2] < 0:
        action[3] = -0.005 * args_cli.sensitivity

    return action


def visualize_images_live(tensor):
    # tensor shape can be (i, height, width) or (i, height, width, channels)
    i = tensor.shape[0]

    if len(tensor.shape) == 3:
        # case when the tensor has no channels dimension (grayscale images)
        tensor = np.expand_dims(
            tensor, -1
        )  # add a channels dimension, becomes (i, height, width, 1)

    # determine if the images are grayscale or color
    channels = tensor.shape[-1]

    if channels == 1:
        # convert grayscale images to 3 channels by repeating the single channel
        tensor = np.repeat(tensor, 3, axis=-1)
        tensor = np.where(np.isinf(tensor), np.nan, tensor)
        # max depth = 10.0m
        tensor = tensor / 10.0
    elif channels == 4:
        # use only the first 3 channels as RGB, ignore the 4th channel (perhaps alpha)
        tensor = tensor[..., :3]
        tensor = tensor / 255.0
    else:
        warnings.warn(f"Unexpected channel number {channels}.", UserWarning)

    # get the height and width from the first image (all images have the same size)
    height, width = tensor.shape[1], tensor.shape[2]

    # calculate the grid size
    cols = int(math.ceil(math.sqrt(i)))
    rows = int(math.ceil(i / cols))

    # create an empty canvas to hold the images
    canvas = np.zeros((rows * height, cols * width, 3))

    for idx in range(i):
        row = idx // cols
        col = idx % cols
        # place the image in the grid cell
        canvas[
            row * height : (row * height) + height,
            col * width : (col * width) + width,
            :,
        ] = tensor[idx]

    # display the canvas
    if not hasattr(visualize_images_live, "img_plot"):
        # create the plot for the first time
        visualize_images_live.fig = plt.figure(figsize=(8, 8))
        visualize_images_live.fig.canvas.manager.set_window_title("Imgs")
        visualize_images_live.img_plot = plt.imshow(canvas)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        # update the existing plot
        visualize_images_live.img_plot.set_data(canvas)

    plt.draw()
    plt.pause(0.001)  # pause to allow the figure to update


def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

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
            
            # if args_cli.task == "":
            #     actions = actions.view(self.num_envs, self.num_drones, -1)

            # apply actions
            obs, _, _, _, _ = env.step(actions)
            visualize_images_live(obs["policy"].cpu().numpy())

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
