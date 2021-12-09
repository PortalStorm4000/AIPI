## Not for online editor ##
from cmu_graphics import *
## Not for online editor ##

#START SCREEN
def initStartScreen():
    # Set the mode to startScreen so the rest of the code can react accordingly
    app.mode = 'startScreen'
    #Create text describing how to play the game and how to start
    app.background = 'lavender'
    Label("Example program. W is up. Check console for text output", 200, 185)
    Label("Press S to start. R to reset", 200, 215)

#FUNCTION TO CREATE/RESET THE GAME SPACE (background, players, enemies, score, etc)
def startGame():
    app.group.clear()
    app.mode = 'playing'

    #ADD GAME OBJECTS (player, enemies, obstacles, pretty much anything you interect with or that moves)
    #Since we will want to use these objects outside of this function's scope, its easiest to just add them t the 'app.' group.
    app.player = RegularPolygon(200, 200, 20, 5, fill='black')


# --- PUT YOUR FUNCTIONS HERE ---
def exampleAlways():
   print('Always saying Hello World')
   
def exampleOnKeyPress(keys):
   if 'w' in keys:
       app.player.top -= 1



#HANDLE INPUTS - returns a list of keys currently pressed EI: ['d','space','enter']
def onKeyHold(keys):
    if app.mode == 'playing':
        exampleOnKeyPress(keys)
    if app.mode == 'startScreen' and 's' in keys:
        startGame()
    if 'r' in keys:
        startGame()

#ONSTEP - Code which runs every step/tick
#stepsPerSecond - How many times the code loops per second, you can set this to any number
app.stepsPerSecond = 30 
def onStep():
    if app.mode == 'playing':
        exampleAlways()

#START THE GAME - After all the required functions have been defined, call the start screen function
initStartScreen()


## Not for online editor ##
cmu_graphics.run()
## Not for online editor ##
