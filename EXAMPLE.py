## Not for online editor ##
from cmu_graphics import *
## Not for online editor ##

#START SCREEN
def initStartScreen():
    # Set the mode to startScreen so the rest of the code can react accordingly
    app.mode = 'startScreen'
    #Create text describing how to play the game and how to start
    app.background = 'lavender'
    Label("An example program.", 200, 185)
    Label("Press S to start", 200, 215)

#FUNCTION TO CREATE/RESET THE GAME SPACE (background, players, enemies, score, etc)
def startGame():
    #Since startGame won't always be called from an empty screen, clear it so we know nothing will conflict
    app.group.clear()
    #Set the app mode to playing so the rest of the code can react accordingly
    app.mode = 'playing'

    #SET BACKGROUND AND ADD SHAPES WHICH MAKE UP THE BACKGROUND
    app.background = 'floralWhite'
    
    #ADD GAME OBJECTS (player, enemies, obstacles, pretty much anything you interect with or that moves)
    #Since we will want to use these objects outside of this function's scope, its easiest to just add them t the 'app.' group
    app.player = RegularPolygon(200, 200, 20, 5, fill='black')
    app.enemy = RegularPolygon(200, 100, 20, 4, fill='red')
    app.goal = RegularPolygon(200, 20, 20, 4, fill='green')
    
#HANDLE PLAYER INPUT/MOVEMENT
def movePlayer(keys):
    #MOVE PLAYER
    #We can define a single speed variable so if we need to change it later, we don't have to change it for each part that uses it
    speed = 5
    
    #Since keys are returned in a 'list, we can use the Python 'in' operator to check through the whole list
    #We can easily move the player up/down/left/right by taking their current position, and then increasing or decreasing it
    if 'w' in keys:
        app.player.top -= speed
    if 's' in keys:
        app.player.top += speed
    if 'a' in keys:
        app.player.left -= speed
    if 'd' in keys:
        app.player.left += speed
        
    #KEEP PLAYER INBOUNDS - Check if the player is out of bounds, and if so, reset them
    if app.player.left < 0:
            app.player.left = 0
    elif app.player.left > 360:
            app.player.left = 360
    if app.player.top < 0:
            app.player.top = 0
    elif app.player.top > 360:
            app.player.top = 360


#HANDLE ENEMY MOVEMENT
direction = 'left'
def moveEnemy():
    global direction
    #If the enemy hits the game boundries, set its direction to the other direction
    if app.enemy.left < 0:
        direction = 'right'
    elif app.enemy.left > 360:
        direction = 'left'

    #If left, go left. If not, go right
    if direction == 'left':
        app.enemy.left -= 5
    else:
        app.enemy.left += 5
    

#CREATE/DISPLAY FAIL SCREEN
def onFail():
    app.mode = 'gameOver'
    app.group.clear()
    Label("You failed, whoops. Better luck next time", 200, 200,)
    Label("Press R to Restart", 200, 220,)

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
        onFail()

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

#ONSTEP - Code which runs every step/tick
#stepsPerSecond - How many times the code loops per second
app.stepsPerSecond = 30 
def onStep():
    #If we aren't playing the game, we don't need to constantly run game logic
    if app.mode == 'playing':
        checkFail()
        checkWin()
        moveEnemy()

#START THE GAME - After all the required functions have been defined, call the start screen function
initStartScreen()


## Not for online editor ##
cmu_graphics.run()
## Not for online editor ##
