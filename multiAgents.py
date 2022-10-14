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

        def get_score_state_at_ghost_state(self, gameState: GameState, current_depth, agent_index):
            # first check if the game is over or if we have reached the depth
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            # we check if we are at the last index, i.e. if the next turn is pacmans
            if agent_index == gameState.getNumAgents() - 1:
                next_agent = 0
            else:
                next_agent = agent_index + 1

            minimizer_score = float('inf')
            ghost_score = float('inf')

            all_future_actions_from_ghosts_state = gameState.getLegalActions(agent_index)

            for each_ghost_action in all_future_actions_from_ghosts_state:
                # this state is if the ghosts have made their moves
                future_ghost_state = gameState.generateSuccessor(agent_index, each_ghost_action)

                # at each state find the agent that will yield the lowest value
                if next_agent >= 1:
                    ghost_score = get_score_state_at_ghost_state(self, future_ghost_state, current_depth, next_agent)
                # means we move on to next depth because pacmans turn is next
                else:
                    # if the next movement is the end of our depth limit return a score
                    if current_depth == self.depth - 1:
                        ghost_score = self.evaluationFunction(future_ghost_state)
                    # otherwise do recursive call on pacman node
                    else:
                        # TODO fix to do
                        ghost_score = get_score_at_pacman_state(self,future_ghost_state,current_depth + 1)

                # this needs to be in the loop
                if ghost_score < minimizer_score:
                    minimizer_score = ghost_score

            return minimizer_score

        def get_score_at_pacman_state(self, gameState: GameState, current_depth):
            # first check if the game is over or if we have reached the depth
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            #initialize
            maximizer_score = float('-inf')
            pac_score = float('-inf')

            all_future_actions_from_pac = gameState.getLegalActions()

            for each_action in all_future_actions_from_pac:
                future_pac_state = gameState.generateSuccessor(0, each_action)

                #not exactly sure why i am not checking if self.depth - 1 instead here
                # please provide feedback TA's if you know
                if current_depth == self.depth:
                    pac_score = self.evaluationFunction(future_pac_state)
                else:
                    pac_score = get_score_state_at_ghost_state(self, future_pac_state,current_depth, 1)

                if pac_score > maximizer_score:
                    maximizer_score = pac_score

            return maximizer_score



        #initialize variables to keep track
        max_value = float('-inf')
        #setting up action to return variable
        returned_action = Directions.STOP
        initial_depth = 0

        #some info to remember
        #there is an agent list, pacman has index value = 0, ghosts have index value >= 1
        # there can be any number of ghosts

        # we start at pacman current state
        # let's check if the game is over
        if gameState.isWin() or gameState.isLose():
            # if the game is over we still need to return an action
            return returned_action

        #first turn is pacmans turn
        #pacman acts as maximizer and wants to pick a state with the largest utility value
        # lets get all potential pacman actions so we can get all future pacman states
        all_future_pacman_actions = gameState.getLegalActions()


        # get each state for corresponding action from pacman state
        for each_action in all_future_pacman_actions:
            # pass in 0 becuase that is pacmans index value
            future_state = gameState.generateSuccessor(0,each_action)
            # this is the state of one pacman action happening

            #in order to calculate score of this state we must call agent recursive function
            #TODO FIX, not just value
            future_state_score = get_score_state_at_ghost_state(self,future_state,initial_depth,1)

            if future_state_score > max_value:
                max_value = future_state_score
                returned_action = each_action

        return returned_action

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
