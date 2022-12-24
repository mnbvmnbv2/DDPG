import numpy as np
import tensorflow as tf

from networks import get_actor, get_critic
from helpers import OUActionNoise


class Agent:
    def __init__(
        self,
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
    ):

        self.continuous = continuous
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        # This is used to make sure we only sample from used buffer space
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        if self.continuous:
            self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        else:
            self.action_buffer = np.zeros((self.buffer_capacity, disc_actions_num))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1)).astype(np.float32)
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon  # Epsilon greedy
        self.loss_func = loss_func

        self.ou_noise = OUActionNoise(
            mean=np.zeros(1),
            std_deviation=float(std_dev) * np.ones(1),
            theta=theta,
            dt=dt,
        )

        self.actor_model = get_actor(
            num_states,
            num_actions,
            continuous,
            disc_actions_num,
            actor_layer_1,
            actor_layer_2,
        )
        self.critic_model = get_critic(
            num_states,
            num_actions,
            continuous,
            disc_actions_num,
            critic_layer_1,
            critic_layer_2,
        )
        self.target_actor = get_actor(
            num_states,
            num_actions,
            continuous,
            disc_actions_num,
            actor_layer_1,
            actor_layer_2,
        )
        self.target_critic = get_critic(
            num_states,
            num_actions,
            continuous,
            disc_actions_num,
            critic_layer_1,
            critic_layer_2,
        )

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=actor_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=adam_actor_eps,
            amsgrad=actor_amsgrad,
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=adam_critic_eps,
            amsgrad=critic_amsgrad,
        )
        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    def record(self, obs_tuple):
        # Reuse the same buffer replacing old entries
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    # Calculation of loss and gradients
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        loss_func,
    ):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + done_batch * self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = loss_func(y, critic_value)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)

        # Gradient clipping
        critic_gvd = zip(critic_grad, self.critic_model.trainable_variables)
        critic_capped_grad = [
            (tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1), var)
            for grad, var in critic_gvd
        ]

        self.critic_optimizer.apply_gradients(critic_capped_grad)

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)

        actor_gvd = zip(actor_grad, self.actor_model.trainable_variables)
        actor_capped_grad = [
            (tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1), var)
            for grad, var in actor_gvd
        ]

        self.actor_optimizer.apply_gradients(actor_capped_grad)

    def learn(self):
        # Sample only valid data
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])

        self.update(
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            self.loss_func,
        )

    def policy(
        self,
        state,
        disc_actions_num=4,
        noise_object=0,
        use_noise=True,
        noise_mult=1,
        rng=np.random.default_rng(1),
    ):
        if use_noise:
            if self.continuous:
                sampled_actions = tf.squeeze(self.actor_model(state))

                noise = noise_object()

                sampled_actions = sampled_actions.numpy() + noise * noise_mult

                # We make sure action is within bounds
                legal_action = np.clip(sampled_actions, -1, 1)
                return [np.squeeze(legal_action)][0]
            else:
                if rng.random() < self.epsilon:
                    action = np.zeros(disc_actions_num)
                    action[np.random.randint(0, disc_actions_num, 1)[0]] = 1
                    return action
                else:
                    return self.actor_model(state)
        else:
            if self.continuous:
                sampled_actions = tf.squeeze(self.actor_model(state)).numpy()
                legal_action = np.clip(sampled_actions, -1, 1)
                return [np.squeeze(legal_action)][0]
            else:
                return self.actor_model(state)
