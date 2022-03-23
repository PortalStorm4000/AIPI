from this import d
from .ai_base import AIBase

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Storage():
    def __init__(self

                # self,
                # target_actor,
                # target_critic,
                # critic_model,
                # critic_optimizer,
                # actor_model,
                # actor_optimizer,
                # gamma
                ):
        self.target_actor = None #target_actor 
        self.target_critic = None #target_critic
        self.critic_model = None #critic_model
        self.critic_optimizer = None #critic_optimizer
        self.actor_model = None#actor_model
        self.vactor_optimizer = None#actor_optimizer
        self.gamma = None #gamma
        

store = Storage()


#Generates noise for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()


    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, num_actions, num_states, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = store.target_actor(next_state_batch, training=True)
            y = reward_batch + store.gamma * store.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = store.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, store.critic_model.trainable_variables)
        store.critic_optimizer.apply_gradients(
            zip(critic_grad, store.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = store.actor_model(state_batch, training=True)
            critic_value = store.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, store.actor_model.trainable_variables)
        store.actor_optimizer.apply_gradients(
            zip(actor_grad, store.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)



class DDPG(AIBase):
    def __init__(self, aiInterface, startGame, num_inputs, num_actions):
        #inherit previous inputs
        super().__init__(aiInterface, startGame, num_inputs, num_actions)

        #Setup env
        self.reset()
        self.prev_state = self.state
        self.episodic_reward = 0

        #Number of inputs and actions/outputs
        self.num_states = num_inputs
        self.num_actions = num_actions

        #Clamping output of network
        self.upper_bound = 400 #400
        self.lower_bound = 0

        #Model
        store.actor_model = self.get_actor()
        store.critic_model = self.get_critic()

        store.target_actor = self.get_actor()
        store.target_critic = self.get_critic()

        # Making the weights equal initially
        store.target_actor.set_weights(store.actor_model.get_weights())
        store.target_critic.set_weights(store.critic_model.get_weights())

        std_dev = 100
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        store.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        store.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.total_episodes = 100
        # Discount factor for future rewards
        store.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

        self.buffer = Buffer(self.num_actions, self.num_states, 50000, 64)


        # To store reward history of each episode
        self.ep_reward_list = []
        # To store average reward history of last few episodes
        self.avg_reward_list = []

    

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        inputs = tf.keras.layers.LayerNormalization(axis=-1, center=True,scale=True,trainable=True,name='input_normalized',)(inputs)#normalization

        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_input = tf.keras.layers.LayerNormalization(axis=-1, center=True,scale=True,trainable=True,name='input_normalized',)(state_input)#normalization

        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(store.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]


    def reset(self):
        self.startGame()
        self.state = self.aiInterface(None)[0] #index gets only first result as aiInterface returns a tuple
        self.episode_reward = 0


    def train_loop(self, goalCompleted):
        if goalCompleted:
            return True

        isRunning = True
        if isRunning: #TODO: CHANGE TO IF

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_state), 0)

            action = self.policy(tf_prev_state, self.ou_noise)

            # Recieve state and reward from environment.
            self.state, reward, done, info = self.aiInterface(action)

            self.buffer.record((self.prev_state, action, reward, self.state))
            self.episodic_reward += reward

            self.buffer.learn()
            self.update_target(store.target_actor.variables, store.actor_model.variables, self.tau)
            self.update_target(store.target_critic.variables, store.critic_model.variables, self.tau)

            # End this episode when `done` is True
            if done:
                isRunning = False

            self.prev_state = self.state

        if not(isRunning):
            self.ep_reward_list.append(self.episodic_reward)
            self.episode_reward = 0

            #log
            # Mean of last 40 episodes
            avg_reward = np.mean(self.ep_reward_list[-40:])
            #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            self.avg_reward_list.append(avg_reward)
            
            self.reset()

            