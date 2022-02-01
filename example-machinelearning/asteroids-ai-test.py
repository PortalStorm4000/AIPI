## Not for online editor ##
from cmu_graphics import *
## Not for online editor ##

#IMPORT MODULES (available to online editor)
import math
import random
import numpy as np
import math
import aipi
import os


#START SCREEN
def initStartScreen():
    # Set the mode to startScreen so the rest of the code can interact correctly
    app.mode = 'startScreen'
    #Create text describing how to play the game and how to start
    app.background = 'black'
    Label("Don't hit the asteroids and get as high a score as you can!", 200, 185, fill='white')
    Label("W,A, and D to move", 200, 200, fill='white')
    Label("Press S to start", 200, 215, fill='white')

#FUNCTION TO CREATE/RESET THE GAME SPACE (background, players, enemies, score, etc)
def startGame():
    #Since startGame won't always be called from an empty screen, clear it so it is empty.
    app.group.clear()
    #Set the mode to playing (the rest of the program needs to know we've started)
    app.mode = 'playing'
    #Set/reset the global time variable to 0 (another function will handle the actual time keeping)
    #(WARNING: Globals should be used sparingly)
    global time
    time = 0
    
    #CREATE BACKGROUND
    app.background = 'black'
    
    #CREATE ASTEROIDS
    #Each individual asteroid is a simple pentagon
    #Their spawn positions are determined by the first two numbers (centerX, centerY),
    #then next their size, then how many sides (5 = pentagon), then their color, and whatever other attributes
    #(Since our background is black and that is the default color for shapes, we need to set the color attribute so we can actually see it)
    #(We set their spawn positions away from the player so they one doesn't accidently spawn inside the player)

    #Since all the asteroids in the game will act the same, we can add them all to a group to
    #detect them all at once or use a For loop to apply a section of code to all of them individually
    app.asteroids = Group(
        RegularPolygon(380, 20, 20, 5, fill='brown'),
        RegularPolygon(20, 20, 20, 5, fill='brown'),
        RegularPolygon(20, 380, 20, 5, fill='brown')
    )
    
    #CREATE PLAYER
    #While you could just use a single triangle, using multiple shapes makes the players ship look cooler.
    #We can use a group again to move all the shapes making up the ship at once
    app.player = Group(
        RegularPolygon(200, 200, 20, 3, fill='white'),
        RegularPolygon(200, 210, 20, 4, fill='white')
    )

    #CREATE SCORE LABEL
    #Let the player see their score. Its added last so it appears ontop of everything else
    app.score = Label(0, 15, 15, size=15, font='montserrat', fill='white')
    

#HANDLE PLAYER INPUT
def handlePlayer(keys):
    #Since keys are returned in a 'list, we can use the Python 'in' operator to check through the whole list
    if('w' in keys):
        #To move the player forwards in the direction they are currently facing, we can use the
        #getPointinDir(x,y,angle,distance) function with the player's coords and how far we want to move as the parameters
        newLeft, newTop = getPointInDir(app.player.left, app.player.top, app.player.rotateAngle, 5)
        #Check that these proposed coords are in bounds. If yes, set the player position to them. If no, do nothing (IE keep current coords and not let them move out of bounds). 
        if(not(newLeft < 0 or newLeft > 360)):
            app.player.left = newLeft
        if(not(newTop < 0 or newTop > 360)):
            app.player.top = newTop
    else:
        #To move the player forwards in the direction they are currently facing, we can use the
        #getPointinDir(x,y,angle,distance) function with the player's coords and how far we want to move as the parameters
        newLeft, newTop = getPointInDir(app.player.left, app.player.top, app.player.rotateAngle, 1)
        #Check that these proposed coords are in bounds. If yes, set the player position to them. If no, do nothing (IE keep current coords and not let them move out of bounds). 
        if(not(newLeft < 0 or newLeft > 360)):
            app.player.left = newLeft
        if(not(newTop < 0 or newTop > 360)):
            app.player.top = newTop
    #Rotate the player on keypress (set the player angle to the current player angle +- the given amount)
    if('d' in keys):
        app.player.rotateAngle += 5
    elif('a' in keys):
        app.player.rotateAngle -= 5    

#HANDLE ASTEROID MOVEMENT AND BOUNCING
def handleAsteroids():
    #To make the asteroids speed up over time to add difficulty,
    #we take the global variable keeping track of time and set the speed proportional to it.
    #We then double check it is at a minimum speed and if not set the speed to the minimum
    global time
    speed = time/127
    if (speed < 1):
        speed = 1

    #Loop through all the asteroids by using a 'for' loop on the asteroids group. 
    for asteroid in app.asteroids:
        #Move the given asteroid forwards using the same getPointInDir method as the player's forwards movement
        asteroid.left, asteroid.top = getPointInDir(asteroid.left, asteroid.top, asteroid.rotateAngle, speed)
        #if a given asteroid is out of bounds, make it face towards the player again with some random variation
        #This makes sure the asteroids are always headed back into bounds to the player, but keeps it from being preditable
        if((asteroid.left < 0 or asteroid.left > 380) or (asteroid.top < 0 or asteroid.top > 380)):
            asteroid.rotateAngle = angleTo(asteroid.left, asteroid.top, app.player.left, app.player.top) + random.randrange(-100, 100)
        
#increases the timer and updates score on each run
def handleScore():
    #Set the global time variable to itself plus 1 every step/tick
    #This will make the time variable equal to [the steprate * seconds]
    #Since our score is based off of seconds, we divide time (steps occured) by the steprate (steps per second) to the seconds.
    global time
    time += 1
    app.score.value = math.floor(time/app.stepsPerSecond)

#Creates game over screen
def onFail():
    app.mode = 'gameOver'
    app.group.clear()
    scoreMessage ='You got through ' + str(app.score.value) + ' seconds, nice!'
    Label(scoreMessage, 200, 200, fill='white')
    Label("Press R to Restart", 200, 220, fill='white')

#Returns true when the player fails
def handleFail():
    #Check for each asteroid in the asteriods group if they touch and of the player parts
    #If yes they touch, set the mode to 'gameOver', clear the app of all the current game objects,
    #and set text describing the score and how to restart
    for asteroid in app.asteroids:
        for playerShape in app.player:
            if(asteroid.hitsShape(playerShape) == True):
                return True


def aiInterface(action):
    if not(action==None): handlePlayer(['w','a','d'][action])
    done = False
    reward = 1
    if handleFail():
        done = True
        reward = 0
    
    #Return our game's state, reward, and if run needs reset back to our NN
    state = [app.player.rotateAngle, app.player.centerX, app.player.centerY]
    for asteroid in app.asteroids:
        state.append(asteroid.centerX)
        state.append(asteroid.centerY)
    return np.array(state, dtype=np.float32), reward, done, {}


ai = aipi.ActorCritic(aiInterface, startGame, 9, 3)



#FUNCTIONS DECLARED BY CMU_GRAPHICS# - We still define them, but we can't change their names or when they are called

#save models on number keypress
def onKeyPress(key):
    for number in range(9):
        number +=1
        if key == str(number):
            location = 'models/'+str(os.path.basename(__file__))+'_'+str(number)
            ai.saveModel(location)

            
#HANDLE INPUTS - returns a list of keys currently pressed EI: ['d','space','enter']
def onKeyHold(keys):
    #Since keys are returned in an unkown list, we can use the Python 'in' operator
    #We also check the mode (to make sure it doesn't accidently restart while playing, or try to move objects while in the menu)
    if(app.mode == 'playing'):
        handlePlayer(keys)
    elif(app.mode == 'startScreen' and 's' in keys):
        startGame()
    elif app.mode == 'gameOver' and 'r' in keys:
        startGame()

#ONSTEP - Code which runs every step/tick
#stepsPerSecond - How many times the code loops per second
app.stepsPerSecond = 3000000000
def onStep():
    #Check if we are in the 'playing' mode, then run all the game functions which aren't already called by a built-in function
    if (app.mode == 'playing'):
        handleAsteroids()
        handleScore()
        ai.train_loop(False) #ML TRAINING
        if handleFail(): onFail()

#After defining all the functions and code, start the game on the start screen
startGame()


## Not for online editor ##
cmu_graphics.run()
## Not for online editor ##
