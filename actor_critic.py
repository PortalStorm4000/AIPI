#Code taken from keras example by Apoorv Nandan. https://github.com/keras-team/keras-io/blob/master/examples/rl/actor_critic_cartpole.py
#Modified to work with games made with the Cmu Graphics library by Jonathan S.

from .ai_base import AIBase

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ActorCritic(AIBase):
    def __init__(self, aiInterface, startGame, num_inputs, num_actions):
        #inherit previous inputs
        super().__init__(aiInterface, startGame, num_inputs, num_actions)
        
        #Create model
        self.gamma = 0.99  # Discount factor for past rewards
        self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        self.reset()

        num_hidden = 128
        inputs = layers.Input(shape=(self.num_inputs,))
        #normalization
        inputs = tf.keras.layers.LayerNormalization(axis=-1, center=True,scale=True,trainable=True,name='input_normalized',)(inputs)
        
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        common = layers.Dense(num_hidden, activation="relu")(common)#second added by me
        self.action = layers.Dense(self.num_actions, activation="softmax")(common)
        self.critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[self.action, self.critic])
        #model = keras.models.load_model('winner_winner')

        #Values for storing actor critic data/history
        self.optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode_count = 0

    def reset(self):
        self.startGame()
        self.state = self.aiInterface(None)[0] #index gets only first result as aiInterface returns a tuple
        self.episode_reward = 0


    #Train
    def train_loop(self, goalCompleted):  # Run until solved
        if goalCompleted:
            return True
        
        isRunning = True
        with tf.GradientTape() as tape:
            if isRunning:
                self.state = tf.convert_to_tensor(self.state)
                self.state = tf.expand_dims(self.state, 0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                self.action_probs, critic_value = self.model(self.state)
                self.critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                action = np.random.choice(self.num_actions, p=np.squeeze(self.action_probs))
                self.action_probs_history.append(tf.math.log(self.action_probs[0, action]))

                # Apply the sampled action in our environment
                self.state, reward, done, _ = self.aiInterface(action)
                self.rewards_history.append(reward)
                self.episode_reward += reward

                if done:
                    isRunning = False


            if isRunning==False:
                # Update running reward to check condition for solving
                self.running_reward = 0.05 * self.episode_reward + (1 - 0.05) * self.running_reward

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in self.rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.action_probs_history, self.critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                self.action_probs_history.clear()
                self.critic_value_history.clear()
                self.rewards_history.clear()

                self.reset()

                # Log details
                self.episode_count += 1
                if self.episode_count % 10 == 0:
                    template = "running reward: {:.2f} at episode {}"
                    print(template.format(self.running_reward, self.episode_count))

                if goalCompleted:  # Condition to consider the task solved
                    print("Solved at episode {}!".format(self.episode_count))
                    #model.save('/odels/winner_winner')
                    #app.mode='win'
                    #onWin()
                    return
