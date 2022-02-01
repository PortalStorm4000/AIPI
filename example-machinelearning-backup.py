## Not for online editor ##
from cmu_graphics import *
## Not for online editor ##

#imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

#FUNCTION TO CREATE/RESET THE GAME SPACE (background, players, enemies, score, etc)
lastDistance = 0
lastY = 200
timeLasted = 0
def startGame():
    #reset playing area (we don't know what exists before now, so we clear it to know its in its preferred state)
    app.group.clear()
    global lastDistance
    lastDistance = 0
    global lastY
    lastY = 200
    global timeLasted
    timeLasted = 0

    #Create playing area
    app.mode = 'playing'
    app.background = 'floralWhite'
    app.player = RegularPolygon(200, 200, 20, 5, fill='black')
    app.enemy = RegularPolygon(200, 100, 20, 4, fill='red')
    app.goal = RegularPolygon(200, 20, 20, 4, fill='green')

#HANDLE PLAYER MOVEMENT (moves player based off of key input)
def movePlayer(keys):
    SPEED = 5 #Using constants lets us easily tune difficulity the game
     
    #Since keys are returned in a 'list, we can use the Python 'in' operator to check through the whole list
    #We can easily move the player up/down/left/right by taking their current position, and then increasing or decreasing it
    if 'w' in keys: app.player.centerY -= SPEED
    if 's' in keys: app.player.centerY += SPEED
    if 'a' in keys: app.player.centerX -= SPEED
    if 'd' in keys: app.player.centerX += SPEED
        
    #KEEP PLAYER INBOUNDS - Check if the player is out of bounds, and if so, move them back in bounds
    if app.player.centerX < 20: app.player.centerX = 20
    if app.player.centerX > 380: app.player.centerX = 380
    if app.player.centerY < 20: app.player.centerY = 20
    if app.player.centerY > 380: app.player.centerY = 380


#HANDLE ENEMY MOVEMENT
direction = 'left'
def moveEnemy():
    ENEMY_SPEED = 5 #Using constants lets us easily tune difficulity the game
    
    #If the enemy hits the game boundries, set its direction to the other direction
    global direction
    if app.enemy.left < 0: direction = 'right'
    elif app.enemy.left > 360: direction = 'left'

    #Just setting a basic variable won't move it, we have to do that ourselves
    if direction == 'left': app.enemy.left -= ENEMY_SPEED
    else: app.enemy.left += ENEMY_SPEED
    
#CREATE/DISPLAY FAIL SCREEN
def onWin():
    app.mode = 'win'
    app.group.clear()
    Label("YAY! YOU GOT IT!", 200, 200)
    Label("Press R to try again!", 200, 220)

#CHECK IF THE PLAYER FAILED
def checkFail():
    #Check if the player hits the enemy and if so, activate the fail scenario
    if app.player.hitsShape(app.enemy):
        startGame()

#CHECK IF THE PLAYER WON
def checkWin():
    #Check if the player reaches the goal. If so, display the win scenario
    if app.player.hitsShape(app.goal):
        onWin()

#FUNCTIONS DECLARED BY CMU_GRAPHICS# - We still define them, but we can't change their names or when they get called
                
#HANDLE INPUTS - returns a list of keys currently pressed EI: ['d','space','enter']
def onKeyHold(keys):
    #Since keys are returned in an unkown list, we can use the Python 'in' operator
    #We also check the mode (to make sure it doesn't accidently restart while playing, or try to move objects while in the menu)
    if app.mode == 'playing':
        movePlayer(keys)
    elif app.mode == 'startScreen' and 's' in keys:
        startGame()
    elif(app.mode == 'gameOver' or app.mode == 'win') and 'r' in keys:
        startGame()

#save models on number keypress
def onKeyPress(key):
    global model
    for number in range(9):
        number +=1
        if key == str(number):
            location = 'models/example_'+str(number)
            model.save(location)

#ONSTEP - Code which runs every step/tick
#stepsPerSecond - How many times the code loops per second
app.stepsPerSecond = 100
def onStep():
    #If we aren't playing the game, we don't need to constantly run game logic
    if app.mode == 'playing':
        checkFail()
        checkWin()
        moveEnemy()
        train_loop() #ML TRAINING

##AI FUNCTIONS##


#can't seem to get the win condition to properly run? Temporary fix
goalCompleted = False

#NN's can't use the keyboard or see the screen, so run its actions and fed it game data 
def aiInterface(action):
    #Map AI outputs to player inputs
    #Our possible actions are provided as a list, and is selected by the AI's 'action' via list indexing
    movePlayer(['w','s','a','d'][action])

    #Calculate reward (distance from player to goal)
    distance = math.dist([app.goal.centerX,app.goal.centerY],[app.player.centerX,app.player.centerY])
    #print("Distance from Goal: " + str(distance))

    #if further, bad. If closer, good
    global lastDistance
    global lastY
    
    reward = 0

    if app.player.centerY < lastY: reward += 1
    elif app.player.centerY > lastY: reward -= 1
    
    if distance < lastDistance: reward += 1
    elif distance > lastDistance: reward -= 1
    
    lastDistance = distance
    lastY = app.player.centerY
    
    global goalCompleted
    #If it hits a wall or dies or whatever, run is done
    done = False
    if app.player.hitsShape(app.enemy):
        done = True
        reward = -1
        
    if app.player.hitsShape(app.goal):
        #goalCompleted = True
        print("Success!")
        done = True
        reward += 5
        
    if app.player.centerX <= 20 or app.player.centerX >= 380 or app.player.centerY <= 20 or app.player.centerY >= 380:
        done = True
        reward -= .5
        
    #Kill it if its taking too long
    global timeLasted
    if timeLasted > 300:
        done = True
        reward = 0
    timeLasted += 1

    #Values to be fed into NN on next loop
    state = (app.player.centerX, app.player.centerY,app.enemy.centerX, app.enemy.centerY)
    
    #Return our state, reward, and if run is done yet back to our NN
    return np.array(state, dtype=np.float32), reward, done, {}

#Create model
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 4
num_hidden = 4

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
#common = layers.Dense(num_hidden, activation="relu")(common)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
#model = keras.models.load_model('winner_winner')

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

#Train
def train_loop():  # Run until solved
    global goalCompleted
    if goalCompleted:
        return True
    
    global state
    global model
    global optimizer
    global huber_loss
    global action_probs_history
    global critic_value_history
    global rewards_history
    global running_reward
    global episode_count
    
    state = np.array((app.player.centerX, app.player.centerY,app.enemy.centerX, app.enemy.centerY), dtype=np.float32)
    episode_reward = 0
    isRunning = True
    with tf.GradientTape() as tape:
        if isRunning:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = aiInterface(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                isRunning = False
                startGame()


        if isRunning==False:
            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
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
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()
        # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))

            if goalCompleted:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                model.save('/odels/winner_winner')
                app.mode='win'
                onWin()
                return

startGame()
## Not for online editor ##
cmu_graphics.run()
## Not for online editor ##
