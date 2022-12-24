import tensorflow as tf
from tensorflow.keras import layers


def get_actor(
    num_states,
    num_actions,
    continuous,
    disc_actions_num,
    layer1,
    layer2,
    init_weights_min=-0.003,
    init_weights_max=0.003,
):

    last_init = tf.random_uniform_initializer(
        minval=init_weights_min, maxval=init_weights_max
    )

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(layer1, activation="relu")(inputs)
    out = layers.LayerNormalization(axis=1)(out)
    out = layers.Dense(layer2, activation="relu")(out)
    out = layers.LayerNormalization(axis=1)(out)
    if continuous:
        outputs = layers.Dense(
            num_actions, activation="tanh", kernel_initializer=last_init
        )(out)
    else:
        outputs = layers.Dense(
            disc_actions_num, activation="softmax", kernel_initializer=last_init
        )(out)

    return tf.keras.Model(inputs, outputs)


def get_critic(num_states, num_actions, continuous, disc_actions_num, layer1, layer2):

    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(64, activation="relu")(state_input)

    if continuous:
        action_input = layers.Input(shape=(num_actions,))
    else:
        action_input = layers.Input(shape=(disc_actions_num,))
    action_out = layers.Dense(64, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(layer1, activation="relu")(concat)
    out = layers.LayerNormalization(axis=1)(out)
    out = layers.Dense(layer2, activation="relu")(out)
    out = layers.LayerNormalization(axis=1)(out)
    outputs = layers.Dense(num_actions)(out)

    return tf.keras.Model([state_input, action_input], outputs)
