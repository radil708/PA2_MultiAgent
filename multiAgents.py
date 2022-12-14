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

                #Is the assumption that the node before the terminal is always minimizer node?
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
            future_state_score = get_score_state_at_ghost_state(self,future_state,initial_depth,1)

            if future_state_score > max_value:
                max_value = future_state_score
                returned_action = each_action

        return returned_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def get_best_minimizer_score_from_minimizer_level(self, gameState: GameState, current_depth: int, current_agent_index, min_alpha_v, min_beta_v):

            display = False

            # first check if the game is over or if we have reached the terminal depth/node
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            #initialize variables to keep track of things
            lowest_minimizer_value = float('inf')
            current_calc_value = None
            pruned_flag = False

            all_future_possible_actions_of_ghost = gameState.getLegalActions(current_agent_index)

            for each_action in all_future_possible_actions_of_ghost:
                future_state = gameState.generateSuccessor(current_agent_index, each_action)

                # this means we will be finished accumulating the ghost movement score for 1 state
                if current_agent_index == gameState.getNumAgents() - 1:
                    # going from minimizer node to maximizer node
                    current_calc_value = get_best_maximizer_score_from_maximizer_level(self,future_state,current_depth + 1,min_alpha_v, min_beta_v)
                else:
                    # this means we are going from minimizer node to minimizer node
                    # this is to accumulate the actions for each ghost on one state
                    current_calc_value = get_best_minimizer_score_from_minimizer_level(
                        self,future_state, current_depth,current_agent_index + 1, min_alpha_v, min_beta_v)

                lowest_minimizer_value = min(lowest_minimizer_value, current_calc_value)
                min_beta_v = min(min_beta_v, lowest_minimizer_value)

                # pruning happens here
                # means that lowest value we can get is already less than the best value
                # for the maximizer above us, cut this off because maximizer will ignore this node already
                # go check out next available node
                #IMPORTANT THIS NEEDS to be less than NOT less than or equal to pass the tests
                if min_beta_v < min_alpha_v:
                    pruned_flag = True
                    break


            if pruned_flag == True:
                if display == True:
                    print(f"pruned at MIN: alpha-1 = {min_alpha_v - 1}")
                    #you watnt the maximizer to ignore this node as well
                    # so return a value 1 lower than the alpha
                    #alpha represents best current option for max
                return min_alpha_v - 1
            else:
                if display == True:
                    print(f"MIN node value = {lowest_minimizer_value}")
                return lowest_minimizer_value

        def get_best_maximizer_score_from_maximizer_level(self, gameState: GameState, current_depth: int, max_alpha_v,max_beta_v):
            display = False

            # first check if the game is over or if we have reached the terminal depth/node
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            largest_maximizer_value = float('-inf')
            current_calc_maximizer_value = float('-inf')
            saved_prev_val = None

            #get all possible actions of maximizer agent/pacman
            all_potential_actions = gameState.getLegalActions(0)

            #one action can generate one state
            # check the best score from each state

            pruned_flag = False

            for each_action in all_potential_actions:

                pruned_flag = False

                future_state = gameState.generateSuccessor(0,each_action)

                #score of current state
                current_calc_maximizer_value = \
                    get_best_minimizer_score_from_minimizer_level(
                        self,future_state, current_depth, 1,max_alpha_v,max_beta_v)

                largest_maximizer_value = max(largest_maximizer_value, current_calc_maximizer_value)
                max_alpha_v = max(max_alpha_v, largest_maximizer_value)

                #This means that the value of this node is alread larger than the minimizer node value above it
                # minimizer node will ignore this node so lets prune
                if max_alpha_v > max_beta_v:
                    pruned_flag = True
                    break

                if pruned_flag == False:
                    saved_prev_val = largest_maximizer_value

            if pruned_flag == True:
                if display == True:
                    print(f"pruned at MAX: {saved_prev_val}")
                # you want minimizer node to ignore this since its already
                # larger than the best lowest option so return a value
                # larger than the best lowest option
                return max_beta_v + 1
            else:
                if display == True:
                    print(f"MAX node val = {largest_maximizer_value}")
                return largest_maximizer_value

        display = False

        # initialize variables
        start_node_value = float('-inf')
        best_action_from_start = Directions.STOP
        start_alpha = float('-inf')
        start_beta = float('inf')
        initial_depth = 0


        # need this var for the for loop to compare and find max score
        indiv_state_calc_value = float('-inf')

        # we start at pacman current state
        # let's check if the game is over
        if gameState.isWin() or gameState.isLose():
            # if the game is over we still need to return an action
            return best_action_from_start

        # now we need to get all actions from start and find the best score among them
        # pacman index is 0 so this is all possible actions from current state of pacman
        all_actions_from_start_node = gameState.getLegalActions(0)

        # generate all states of pacman as a result of an action
        future_states_list = []

        for each_action in all_actions_from_start_node:
            # get the new state as the result of a pacman action
            future_states_list.append(gameState.generateSuccessor(0, each_action))
            # this state is incomplete since it doesn't take into account the actions of the ghost agents

        # lets get the score calculated from the min node

        for each_transition_state_index in range(len(future_states_list)):
            indiv_state_calc_value = \
                get_best_minimizer_score_from_minimizer_level(
                    self,future_states_list[each_transition_state_index],
                    initial_depth,1,start_alpha,start_beta)

            # since start is a maximizer node we need to get a value that is the largest
            if indiv_state_calc_value > start_node_value:
                start_node_value = indiv_state_calc_value
                # update the best action to return
                best_action_from_start = all_actions_from_start_node[each_transition_state_index]

            start_alpha = max(start_alpha, indiv_state_calc_value)
            if display:
                print(f"Current start alpha value = {start_alpha}")
                print("==========================\n")

        return best_action_from_start






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

        #no pruning
        # maximizer still wants to get the largets value
        # expectimax agent will calculate expected value (sum of (child node value * probability of picking child node))
        # prompt says uniform distribution, so every child node has equal change of being chosen

        #minimizer nodes are replaced with chance nodes
        # chance node value represents the average value of all nodes below it
        # chance nodes are the ghosts turn, its possible that the child of chance node can be a chance node
            # ie. one ghost move, then take into account the other ghost move in one turn
        def get_expected_node_value(self, gameState: GameState, current_depth: int, current_agent_index):
            display = False

            # first check if the game is over or if we have reached the terminal depth/node
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            #initialize variables to keep track of things
            current_calc_value = None

            all_future_possible_actions_of_ghost = gameState.getLegalActions(current_agent_index)

            num_possible_actions = len(all_future_possible_actions_of_ghost)
            sum_scores = []

            for each_action in all_future_possible_actions_of_ghost:
                future_state = gameState.generateSuccessor(current_agent_index, each_action)

                # this means we will be finished accumulating the ghost movement score for 1 state
                if current_agent_index == gameState.getNumAgents() - 1:
                    current_calc_value = get_best_maximizer_score_from_maximizer_level(self,future_state,current_depth + 1)
                else:
                    # this means we are going from minimizer node to minimizer node
                    # this is to accumulate the actions for each ghost on one state
                    current_calc_value = get_expected_node_value(
                        self,future_state, current_depth,current_agent_index + 1)


                sum_scores.append(current_calc_value)

            # uniform distribution so just get average value of child nodes
            return sum(sum_scores)/num_possible_actions



        #maximizer still wants to pick largest value, it doesn't rely on probability
        def get_best_maximizer_score_from_maximizer_level(self, gameState: GameState, current_depth: int):
            display = False

            # first check if the game is over or if we have reached the terminal depth/node
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluationFunction(gameState)


            largest_value = float('-inf')

            #get all possible actions of maximizer agent/pacman
            all_potential_actions = gameState.getLegalActions(0)

            #one action can generate one state
            # check the best score from each state

            for each_action in all_potential_actions:

                future_state = gameState.generateSuccessor(0,each_action)

                #score of current state, # max has made a move so now ghosts turn
                current_calc_maximizer_value = \
                    get_expected_node_value(
                        self,future_state, current_depth, 1)

                largest_value = max(largest_value,current_calc_maximizer_value)

            return largest_value



        display = False

        # initialize variables
        start_node_value = float('-inf')
        best_action_from_start = Directions.STOP
        initial_depth = 0


        # need this var for the for loop to compare and find max score
        indiv_state_calc_value = float('-inf')

        # we start at pacman current state
        # let's check if the game is over
        if gameState.isWin() or gameState.isLose():
            # if the game is over we still need to return an action
            return best_action_from_start

        # now we need to get all actions from start and find the best score among them
        # pacman index is 0 so this is all possible actions from current state of pacman
        all_actions_from_start_node = gameState.getLegalActions(0)

        # generate all states of pacman as a result of an action
        future_states_list = []

        for each_action in all_actions_from_start_node:
            # get the new state as the result of a pacman action
            future_states_list.append(gameState.generateSuccessor(0, each_action))
            # this state is incomplete since it doesn't take into account the actions of the ghost agents

        # lets get the score calculated from the min node

        for each_transition_state_index in range(len(future_states_list)):
            indiv_state_calc_value = \
                get_expected_node_value(
                    self,future_states_list[each_transition_state_index],
                    initial_depth,1)

            # since start is a maximizer node we need to get a value that is the largest
            if indiv_state_calc_value > start_node_value:
                start_node_value = indiv_state_calc_value
                # update the best action to return
                best_action_from_start = all_actions_from_start_node[each_transition_state_index]

        return best_action_from_start

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I will approach it similarly to q1
    1.) states that get pacman closer to the closest food pellet should have a higher value
    2.) Apply penalty based on the distance from pacman to ghosts
        A state where pacman is closer to the ghosts will have a higher penalty than
        one where pacman is farther away from ghosts
    """
    "*** YOUR CODE HERE ***"
    # this doesn't take an action so solely based on current state
    # Useful information you can extract from a GameState (pacman.py)
    pacman_position = currentGameState.getPacmanPosition()
    food_positions = currentGameState.getFood().asList()
    power_pellets_position_list = currentGameState.getCapsules()
    ghostAgents = currentGameState.getGhostStates()


    #ghost.scaredTimer gets you turns left that ghost is scared
    utility_value = 0
    closest_dist = float('inf')

    #contribute food bonus
    for food_pos in food_positions:
        calc_dist = util.manhattanDistance(food_pos,pacman_position)
        closest_dist = min(closest_dist, calc_dist)

        if closest_dist <= 1:
            closest_dist = 1

    #get accumulation of ghost distances
    dist_ghosts = 0

    for ghost in ghostAgents:
        dist_to_ghost = util.manhattanDistance(ghost.getPosition(), pacman_position)

        if dist_to_ghost <= 1:
            dist_ghosts += 1
        else:
            dist_ghosts += dist_to_ghost

    #current game score will decrease if moving to a spot with no food, should help
    # give pacman incentive to go towards food
    utility_value += (1/closest_dist) - (1/(dist_ghosts ** 2)) + currentGameState.getScore()

    return utility_value

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
