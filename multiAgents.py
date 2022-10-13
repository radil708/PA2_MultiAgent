# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood() #2D array, True if food, false if not
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        display = False
        #TODO delete print statements

        #SET TO FALSE before submission to remove print statement
        if display is True:
            print(f"new position of pacman: {newPos}") # positions are row, col
            print(f"remaining food {newFood.asList()}") # looks like values are column,row for any element
            print(newFood)
            print(f"new scared times {newScaredTimes}")
            print(f"ghost position {successorGameState.getGhostPositions()}") # position is row, col
            print("=====================================\n")

        #need to define a utility function, higher score is better
        # function should have a lower value if pacman gets close to a ghost
        #function should have a really low value if pacman wants to move where
            # a ghost is
        # function should have a higher value if pacman gets closer to food
        # function should have a really high value if position is position of a food

        #simple idea assign +1 value if new position is in the food list
        #declaring util variable
        utility_value = 0


        # if the action leads to a state without food
        if (currentGameState.getFood().count() == len(newFood.asList())):
            #the states utility should be higher if its close to a food pellet

            # initialize dist var
            closest_dist = float('inf')

            for food_pos in newFood.asList():
                calc_dist = util.manhattanDistance(newPos, food_pos)
                if calc_dist < closest_dist:
                    closest_dist = calc_dist

            if closest_dist == float('inf'):
                raise RuntimeError("Error: Closest food pellet distance can NOT "
                                   "be INFINITY")

            # I am using a 100 scale and sort of assigning arbitrary increase
            # I figure that if the new position is right on a new food position
            # then the utility should increase by 100. If the newpos is 1 unit
            # away from the food pellet then (4/4 - 1/4)100 = 75 etc..

            if closest_dist == 1:
                utility_value += 75
            else:
                utility_value += (1/closest_dist) * 100
        else:
            # the action that lead to successor state must have eaten a pellet
            utility_value += 100

        #apply penalty if moving towards ghost
        for each_ghost in newGhostStates:
            ghost_distance = util.manhattanDistance(newPos, each_ghost.getPosition())

            if ghost_distance == 0:
                ghost_penalty = 75
            elif ghost_distance == 1:
                ghost_penalty = 50
            else:
                ghost_penalty = (1/ghost_distance ** 2) * 100
            utility_value -= ghost_penalty

        return utility_value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        '''
        In minimax, maximizer (pacman in this case) wants to pick state with 
        largest possible utility
        Ghosts want to limit our moves to the smallest possible max value we can pick
        '''

        def minimax(self,gameState: GameState, depth : int, agent_index : int):
            # check if the game is over
            if gameState.isWin() or gameState.isLose():
                # this is the score of the state
                return self.evaluationFunction(gameState)

            # pacman has agent_index = 0
            # pacman is maximizer, he wants to maximize utility value
            if agent_index == 0:
                # get all pacman actions
                all_future_pacman_actions = gameState.getLegalPacmanActions()

                #make a list of all potential scores of each potential future state
                list_all_scores = []

                for each_action in all_future_pacman_actions:
                    future_state = gameState.generatePacmanSuccessor(each_action)
                    # pass in the future state, the depth, and agent index of 1 since
                    # ghosts have agent index of >= 1
                    score = minimax(self, future_state, depth, 1)
                    list_all_scores.append(score)

                return max(list_all_scores)
            # here is the ghost
            else:
                #





        def ghost_action(self, gameState: GameState, depth, agent_index):
            #check if the game is over
            if gameState.isWin() or gameState.isLose():
                # this is the score of the state
                return self.evaluationFunction(gameState)

            #game is not over pacman has moved, now its the ghosts turn

            # get all ghost actions


            x = 5


        #need to keep track of depth
        depth = 0

        # check if the game is over, if over return no action i.e. stop
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP

        # starting at pacman lets generate all possible new states by getting all possible actions
        all_future_pacman_actions = gameState.getLegalPacmanActions()

        for action in all_future_pacman_actions:
            #this only takes into account pacmans move
            future_state = gameState.generatePacmanSuccessor(action)

            # if the action creates a state where we win, then getAction should return that action
            if future_state.isWin():
                return action

            # pacman will choose a state with the largest utility
            #in order to do that we must determine all the potentual actions of the ghost
            future_state_utility_value = ghost_action(self, future_state, depth + 1, 1)




        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
