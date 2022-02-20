## Not for online editor ##
from cmu_graphics import *
## Not for online editor ##

#imports
import numpy as np
import math
import os
import aipi


####-GLOBALS-####
#globals for nn
lastDistance = 0
lastY = 200
timeLasted = 0
successCount = 0
#


####-GAME LOGIC CODE-#####

#FUNCTION TO CREATE/RESET THE GAME SPACE (background, players, enemies, score, etc)

def startGame():
    #reset playing area/data (we don't know what exists before now, so we clear it to know its in its preferred state)
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


####-NEURAL NETWORK CODE-#####

#can't seem to get the win condition to properly run? Temporary fix
goalCompleted = False

#NN's can't use the keyboard or see the screen, so run its actions and fed it game data 
def aiInterface(action):
    #Map AI outputs to player inputs
    #Our possible actions are provided as a list, and is selected by the AI's 'action' via list indexing
    if not(action == None): movePlayer(['w','s','a','d'][action])

    #Calculate reward (distance from player to goal)
    distance = math.dist([app.goal.centerX,app.goal.centerY],[app.player.centerX,app.player.centerY])

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

    #zero minim reward
    if reward < 0: reward = 0
        
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


#CREATE AI OBJECT
#Now that we have set up our interface, create an ai object to so we have ai (aipi module lets us easily use different types of algorthims without making it ourselves)
ai = aipi.DeepQ(aiInterface, startGame, 4, 4)


####-CMU FUNCTIONS/ENGINE LOOP-#####

#ONSTEP - Code which runs every step/tick
#stepsPerSecond - How many times the code loops per second
app.stepsPerSecond = 1000000000
def onStep():
    #If we aren't playing the game, we don't need to constantly run game logic
    if app.mode == 'playing':
        global goalCompleted
        checkFail()
        checkWin()
        moveEnemy()
        ai.train_loop(goalCompleted) #ML TRAINING

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
    for number in range(9):
        number +=1
        if key == str(number):
            location = 'models/'+str(os.path.basename(__file__))+'_'+str(number)
            ai.saveModel(location)

startGame()
## Not for online editor ##
cmu_graphics.run()
## Not for online editor ##
