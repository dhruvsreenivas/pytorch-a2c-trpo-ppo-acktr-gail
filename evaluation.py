import numpy as np
import torch
from a2c_ppo_acktr.utils import get_render_func
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, env_name, seed, num_processes, eval_log_dir, device, render=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Observation reward and next observation
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

        if render:
            get_render_func(eval_envs)()

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
