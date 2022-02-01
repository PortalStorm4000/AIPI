#parameters all aipi ai's use
class AIBase():
    def __init__(self, aiInterface, startGame, num_inputs, num_actions):
        #Map parameters to variables 
        self.aiInterface = aiInterface
        self.startGame = startGame
        self.num_inputs = num_inputs
        self.num_actions = num_actions
