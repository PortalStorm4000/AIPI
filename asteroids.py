## Not for online editor ##
from cmu_graphics import *
## Not for online editor ##

#IMPORT MODULES (available to online editor)
import math
import random

#FUNCTION TO CREATE/RESET THE GAME SPACE (background, players, enemies, score, etc)
def startGame():
    #Since startGame won't always be called from an empty screen, clear it so it is empty.
    app.group.clear()
    #Set the mode to playing (the rest of the program needs to know we've started)
    app.mode = 'playing'
    #make a global variable called time (so it can be used throughout the app) (WARNING: Globals should be used sparingly)
    #Then make sure it is reset/set to 0
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
    #Let the player see their score. Is added last so it appears ontop of everything else
    app.score = Label(0, 15, 15, size=15, font='montserrat', fill='white')
    

#HANDLE PLAYER INPUT
def handlePlayer(keys):
    if('w' in keys):
        newLeft, newTop = getPointInDir(app.player.left, app.player.top, app.player.rotateAngle, 5)
        if(not(newLeft < 0 or newLeft > 360)):
            app.player.left = newLeft
        if(not(newTop < 0 or newTop > 360)):
            app.player.top = newTop
    if('d' in keys):
        app.player.rotateAngle += 5
    elif('a' in keys):
        app.player.rotateAngle -= 5    

#HANDLE ASTEROID MOVEMENT AND BOUNCING
def handleAsteroids():
    global time
    speed = time/127
    if (speed < 1):
        speed = 1
        
    for asteroid in app.asteroids:
        asteroid.left, asteroid.top = getPointInDir(asteroid.left, asteroid.top, asteroid.rotateAngle, speed)
        if((asteroid.left < 0 or asteroid.left > 380) or (asteroid.top < 0 or asteroid.top > 380)):
            asteroid.rotateAngle = angleTo(asteroid.left, asteroid.top, app.player.left, app.player.top) + random.randrange(-100, 100)
        
#HANDLE TIMER AND SCORE
def handleScore():
    global time
    time += 1
    app.score.value = math.floor(time/app.stepsPerSecond)

#HANDLE FAIL CONDITION
def handleFail():
    for asteroid in app.asteroids:
        #if(asteroids.hitTest(player.centerX, player.centerY) != None):
        for playerShape in app.player:
            if(asteroid.hitsShape(playerShape) == True):
                app.mode = 'gameOver'
                app.group.clear()
                scoreMessage ='You got through ' + str(app.score.value) + ' seconds, nice!'
                Label(scoreMessage, 200, 200, fill='white')
                Label("Press R to Restart", 200, 220, fill='white')

#HANDLE INPUTS - call functions on keypresses/pass keypresses to required functions
#While we define this function, it is called by the cmu_graphics module 
def onKeyHold(keys):
    if(app.mode == 'playing'):
        handlePlayer(keys)
    elif(app.mode == 'startScreen' and 's' in keys):
        startGame()
    elif app.mode == 'gameOver' and 'r' in keys:
        startGame()
        
app.stepsPerSecond = 30 #Using 30 as it just felt right.
def onStep():
    if (app.mode == 'playing'):
        handleAsteroids()
        handleScore()
        handleFail()

def initStartScreen():
    # Set the mode to startScreen so the game can handle starting correctly
    app.mode = 'startScreen'
    #Create text describing how to play the game and how to start
    app.background = 'black'
    Label("Don't hit the asteroids and get as high a score as you can!", 200, 185, fill='white')
    Label("W,A, and D to move", 200, 200, fill='white')
    Label("Press S to start", 200, 215, fill='white')

initStartScreen()


## Not for online editor ##
cmu_graphics.run()
## Not for online editor ##
