import os
import time
import datetime

import numpy as np
import tensorflow as tf
from gym import spaces
from tensorflow.keras import losses
import matplotlib.pyplot as plt

from helpers import fixed
from agent import Agent


def run(
    env,
    total_trials=1,
    total_episodes=100,
    buffer_capacity=50000,
    batch_size=64,
    std_dev=0.3,
    critic_lr=0.003,
    render=False,
    actor_lr=0.002,
    gamma=0.99,
    tau=0.005,
    noise_mult=1,
    save_weights=True,
    directory="Weights/",
    gamma_func=fixed,
    tau_func=fixed,
    critic_lr_func=fixed,
    actor_lr_func=fixed,
    noise_mult_func=fixed,
    std_dev_func=fixed,
    mean_number=20,
    output=True,
    return_rewards=False,
    total_time=True,
    reward_mod=False,
    solved=200,
    continuous=True,
    seed=1453,
    start_steps=0,
    epsilon=0.2,
    epsilon_func=fixed,
    adam_critic_eps=1e-07,
    adam_actor_eps=1e-07,
    actor_amsgrad=False,
    critic_amsgrad=False,
    actor_layer_1=256,
    actor_layer_2=256,
    critic_layer_1=256,
    critic_layer_2=256,
    theta=0.15,
    dt=1e-2,
    disc_actions_num=4,
    loss_func=losses.MeanAbsoluteError(),
    use_gpu=True,
):

    tot_time = time.time()

    _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    try:
        continuous = env.continuous
    except:
        continuous = True

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    num_states = env.observation_space.low.shape[0]
    if continuous:
        num_actions = env.action_space.shape[0]
    else:
        num_actions = 1

    # Normalize action space according to https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    env.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype="float32")

    ep_reward_list = []
    avg_reward_list = []
    true_reward_list = []
    true_avg_reward_list = []

    for trial in range(total_trials):
        step = 0

        # Add sublists for each trial
        avg_reward_list.append([])
        ep_reward_list.append([])
        true_reward_list.append([])
        true_avg_reward_list.append([])

        agent = Agent(
            num_states,
            num_actions,
            continuous,
            buffer_capacity,
            batch_size,
            std_dev,
            actor_lr,
            critic_lr,
            gamma,
            tau,
            epsilon,
            adam_critic_eps,
            adam_actor_eps,
            actor_amsgrad,
            critic_amsgrad,
            actor_layer_1,
            actor_layer_2,
            critic_layer_1,
            critic_layer_2,
            theta,
            dt,
            disc_actions_num,
            loss_func,
        )

        for ep in range(total_episodes):
            before = time.time()

            agent.gamma = gamma_func(agent.gamma, ep)
            agent.tau = tau_func(agent.tau, ep)
            agent.critic_lr = critic_lr_func(agent.critic_lr, ep)
            agent.actor_lr = actor_lr_func(agent.actor_lr, ep)
            agent.std_dev = std_dev_func(agent.std_dev, ep)
            agent.epsilon = epsilon_func(agent.epsilon, ep)
            noise_mult = noise_mult_func(noise_mult, ep)

            prev_state = env.reset()
            episodic_reward = 0
            true_reward = 0

            while True:
                if render:
                    env.render()

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                if step >= start_steps:
                    action = agent.policy(
                        state=tf_prev_state,
                        disc_actions_num=disc_actions_num,
                        noise_object=agent.ou_noise,
                        noise_mult=noise_mult,
                        rng=rng,
                    )
                else:
                    action = env.action_space.sample()

                step += 1

                if continuous:
                    try:
                        len(action)
                    except:
                        action = [action]
                    state, reward, done, _ = env.step(action)
                else:
                    state, reward, done, _ = env.step(np.argmax(action))

                true_reward += reward

                # Reward modification
                if reward_mod:
                    reward -= abs(state[0])

                terminal_state = int(not done)

                agent.record((prev_state, action, reward, state, terminal_state))

                agent.learn()

                episodic_reward += reward

                if done:
                    break

                prev_state = state

            ep_reward_list[trial].append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[trial][-mean_number:])
            avg_reward_list[trial].append(avg_reward)
            true_reward_list[trial].append(true_reward)
            true_avg_reward = np.mean(true_reward_list[trial][-mean_number:])
            true_avg_reward_list[trial].append(true_avg_reward)

            if output:
                time_elapsed = time.time() - before
                print(
                    f"Ep {ep:.3f} * "
                    f"AvgReward {avg_reward:.2f} * "
                    f"true AvgReward {true_avg_reward:.2f} * "
                    f"Reward {episodic_reward:.2f} * "
                    f"True Reward {true_reward:.2f} * "
                    f"time {time_elapsed:.2f} * "
                    f"step {step}"
                )

            # Stop if avg is above 'solved'
            if true_avg_reward >= solved:
                break

        # Save weights
        now = datetime.datetime.now()
        timestamp = (
            f"{now.year}.{now.month}.{now.day}.{now.hour}.{now.minute}.{now.second}"
        )
        save_name = f"{env.spec.id}_{continuous}_{timestamp}"
        if save_weights:
            try:
                agent.actor_model.save_weights(
                    f"{directory}actor-trial{trial}_{save_name}.h5"
                )
            except:
                print("actor save fail")
            try:
                agent.critic_model.save_weights(
                    f"{directory}critic-trial{trial}_{save_name}.h5"
                )
            except:
                print("critic save fail")

    # Plotting graph
    for idx, p in enumerate(true_avg_reward_list):
        plt.plot(p, label=str(idx))
    plt.xlabel("Episode")
    plt.ylabel("True Avg. Epsiodic Reward (" + str(mean_number) + ")")
    plt.legend()
    try:
        plt.savefig(f"Graphs/{save_name}.png")
    except:
        print("fig save fail")
    plt.show()

    total_elapsed_time = time.time() - tot_time
    print(f"total time: {total_elapsed_time}s")

    if return_rewards:
        return true_reward_list


def test(
    env, actor_weights, total_episodes=10, render=False, disc_actions_num=4, seed=1453
):
    rewards = []

    _ = env.reset(seed=seed)

    try:
        continuous = env.continuous
    except:
        continuous = True

    num_states = env.observation_space.low.shape[0]
    if continuous:
        num_actions = env.action_space.shape[0]
    else:
        num_actions = 1

    # Normalize action space according to https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    env.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype="float32")

    for _ in range(total_episodes):
        ep_reward = 0

        before = time.time()

        prev_state = env.reset()
        agent = Agent(
            num_states=num_states,
            num_actions=num_actions,
            continuous=continuous,
            buffer_capacity=0,
            batch_size=0,
        )
        agent.actor_model.load_weights(actor_weights)

        while True:
            if render:
                env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = agent.policy(
                state=tf_prev_state, disc_actions_num=disc_actions_num, use_noise=False
            )

            if continuous:
                try:
                    len(action)
                except:
                    action = [action]
                state, reward, done, _ = env.step(action)
            else:
                state, reward, done, _ = env.step(np.argmax(action))

            ep_reward += reward

            if done:
                print(str(time.time() - before) + "s")
                rewards.append(ep_reward)
                break

            prev_state = state

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("True reward")
    plt.show()


def random(env, total_episodes=10, render=False, seed=1453):
    rewards = []

    _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    for _ in range(total_episodes):
        ep_reward = 0

        before = time.time()

        _ = env.reset()

        while True:
            if render:
                env.render()
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            ep_reward += reward

            if done:
                elapsed_time = time.time() - before
                print(f"{elapsed_time}s")
                rewards.append(ep_reward)
                break

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("True reward")
    plt.show()
