from tensorflow.keras import losses

from runs import run
from helpers import fixed


if "__name__" == "main":

    run(
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
    )
