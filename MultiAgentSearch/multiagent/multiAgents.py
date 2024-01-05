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
import random
import util
import math
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

    def __init__(self, index=0):
        self.index = index
        self.cache = {}

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        currentFood = currentGameState.getFood()
        # Initialize the heuristic value to zero
        heuristicValue = 0
        # Check if the game is over
        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")
        # Compute the number of food pellets in the current and successor states
        numCurrentFood = currentFood.count()
        numSuccessorFood = newFood.count()
        min_food_distance = float("inf")
        sum_ghost_distances = 0
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentGhostPositions = currentGameState.getGhostPositions()
        # Compute the distance to the closest food pellet
        for food in newFood.asList():
            min_food_distance = min(min_food_distance, manhattanDistance(newPos, food))

        # Compute the distance to the closest ghost
        for ghost in newGhostStates:
            sum_ghost_distances += manhattanDistance(newPos, ghost.getPosition())

        # Compute the heuristic value
        score = 88 * (currentGameState.getNumFood() - successorGameState.getNumFood()) + 89 * (
                    1 / min_food_distance) + -96 * (1 / sum_ghost_distances) + 70 * sum(newScaredTimes) + -60 * (
                    1 if newPos in currentGhostPositions else 0) + (-1000 if action == Directions.STOP else 0)
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):  # minimax agent

    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, state, currentDepth, agentIndex):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.minimax(successor, successorIndex, successorDepth)[0]

            if successorValue > v:
                if action == Directions.STOP:
                    x = (self.temperature - 0.5) * 12
                    s = 1 / (1 + math.exp(-x))
                    reduce_amount = s * 0.9 + 0.1
                    successorValue *= reduce_amount
                    if successorValue > v:
                        v = successorValue
                        bestAction = action
                else:
                    v = successorValue
                    bestAction = action
        return v, bestAction

    def minValue(self, state, currentDepth, agentIndex):
        v = float("inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.minimax(successor, successorIndex, successorDepth)[0]

            if successorValue < v:
                v = successorValue
                bestAction = action
        return v, bestAction

    def getAction(self, gameState: GameState):
        """
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

        bestScore, bestAction = self.minimax(gameState, 0, 0)

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue > v:
                if action == Directions.STOP:
                    x = (self.temperature - 0.5) * 12
                    s = 1 / (1 + math.exp(-x))
                    reduce_amount = s * 0.9 + 0.1
                    successorValue *= reduce_amount
                    if successorValue > v:
                        v = successorValue
                        bestAction = action
                else:
                    v = successorValue
                    bestAction = action

            if v > beta:
                return v, bestAction

            alpha = max(alpha, v)
        return v, bestAction

    def minValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = float("inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue < v:
                v = successorValue
                bestAction = action

            if v < alpha:
                return v, bestAction

            beta = min(beta, v)
        return v, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        self.temperature -= 0.005
        if self.temperature < 0.01:
            self.temperature = 0.01
        bestScore, bestAction = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))

        return bestAction


class ExpectimaxAlphaBetaPruningAgent(MultiAgentSearchAgent):
    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.expValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue > v:
                if action == Directions.STOP:
                    x = (self.temperature - 0.5) * 12
                    s = 1 / (1 + math.exp(-x))
                    reduce_amount = s * 0.9 + 0.1
                    successorValue *= reduce_amount
                    if successorValue > v:
                        v = successorValue
                        bestAction = action
                else:
                    v = successorValue
                    bestAction = action

            if v > beta:
                return v, bestAction

            alpha = max(alpha, v)
        return v, bestAction

    def expValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = 0
        bestAction = None
        allActions = state.getLegalActions(agentIndex)

        if len(allActions) == 0:
            return self.evaluationFunction(state), None
        successorProb = 1 / len(allActions)

        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            v += successorValue
        v /= len(allActions)
        return v, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        bestScore, bestAction = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))

        return bestAction


def aStar(gameState: GameState, goal: tuple, heuristic: callable):
    """
    A* algorithm
    """
    start = gameState.getPacmanPosition()
    frontier = util.PriorityQueue()
    frontier.push((start, []), 0)
    explored = set()

    while not frontier.isEmpty():
        current, path = frontier.pop()
        if current == goal:
            return path
        if current not in explored:
            explored.add(current)
            for next in gameState.getLegalActions():
                successor = gameState.generateSuccessor(0, next)
                nextPos = successor.getPacmanPosition()
                if nextPos not in explored:
                    newPath = path + [next]
                    newCost = len(newPath) + heuristic(nextPos, goal)
                    frontier.push((nextPos, newPath), newCost)
    return []


def euclideanDistance(point1: tuple, point2: tuple):
    """
    Euclidean distance between two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    First, we extract features from the current game state. Those features are:

    - feat_FoodCount: the number of food left on the board

    - feat_DClosestFood: the distance to the closest food
        - Here we use the foodHeuristic function to actually calculate the distance to the closest food using manhattan distance
        - We also check if there is a ghost nearby, if so we return a very high number to avoid it, meaning that no need to risk going for the close food if there is a ghost nearby
        - The returned value is normalized by the maximum distance possible on the board ((maximum_distance - closest_food) / maximum_distance)
        - Then we use math.exp(feat_DClosestFood) to make changes in the distance more sensitive

    - feat_currentScore: the current score of the game

    - feat_isNearGhost: a boolean value (0 or 1) indicating if there is a ghost nearby
        - It works by first using manhattan distance to check if there is a CHANCE that there is a ghost nearby
        - If there is a chance, we use A* to validate that indeed there is a ghost nearby


    Then, we use the weights to calculate the score of the current game state.

    We used a Genetic Algorithm to find the best weights for the features.
        - How it works:
            - We start with a random population of weights, each chromosome is a list of 4 weights [w1, w2, w3, w4] where wi is a random number between -2000 and 2000
            - We then calculate the fitness of each chromosome by running the game 1 time and returning the final game score as the fitness of that chromosome
            - We used ranking selection to select the best chromosomes to be the parents of the next generation
            - We used 80% probabilty for crossover and 35% probabilty for mutation
            - We keep 2 elites from each generation to speedup convergence and not lose valuable weights

        - How to run the GA:
            - First
                ```py
                pip install requirements.txt
                ```
            - Then run
                ```py
                python genetic_algorithm.py -l smallClassic -p AlphaBetaAgent -k 10
                ```

            - l is the layout of the game
            - p is the agent to use
            - k is the number of ghosts

    We multiply each feature by its weight and sum them up to get the score of the current game state
    then return that score
    """

    "*** YOUR CODE HERE ***"

    solution = [
        1687,
        469,
        1040,
        -748,
    ]

    def DClosestFood(current_pos, foodGrid, ghosts_pos):

        # if there is chance (thus manhattanDistance and not exact distance)
        # that there is a ghost nearby dont risk it
        for ghost in ghosts_pos:
            if manhattanDistance(current_pos, ghost) <= 1:
                return 99999

        closestFood = foodHeuristic(current_pos, foodGrid)
        if closestFood == 0:
            closestFood = 1
        return closestFood

    def isNearGhost(current_pos, ghosts_states):
        # exact distance to ghost
        for ghost_state in ghosts_states:
            if ghost_state.scaredTimer == 0:
                estimadedDistance = euclideanDistance(current_pos, ghost_state.getPosition())
                if estimadedDistance <= 1:
                    if len(aStar(currentGameState, ghost_state.getPosition(), manhattanDistance)) <= 1:
                        return 1
            else:
                estimadedDistance = euclideanDistance(current_pos, ghost_state.getPosition())
                if estimadedDistance <= ghost_state.scaredTimer:
                    return -10
        return 0

    current_pos = currentGameState.getPacmanPosition()
    ghosts_pos = currentGameState.getGhostPositions()
    ghosts_states = currentGameState.getGhostStates()

    foodGrid = currentGameState.getFood()
    capsuleList = currentGameState.getCapsules()

    feat_isNearGhost = isNearGhost(current_pos, ghosts_states)

    maximum_distance = currentGameState.data.layout.width + currentGameState.data.layout.height
    closest_food = DClosestFood(current_pos, foodGrid, ghosts_pos)
    # normalize base on maximum distance
    feat_DClosestFood = (maximum_distance - closest_food) / maximum_distance
    # use Exp to make it more sensitive
    feat_DClosestFood = math.exp(feat_DClosestFood)
    feat_currentScore = currentGameState.getScore()
    feat_FoodCount = 1.0 / (len(foodGrid.asList()) + 1)

    features = [feat_currentScore,
                feat_FoodCount,
                feat_DClosestFood,
                feat_isNearGhost]

    score = 0
    for i in range(len(features)):
        score += features[i] * solution[i]
    return score


# Abbreviation
better = betterEvaluationFunction


def foodHeuristic(position, foodGrid):
    food_list = foodGrid.asList()

    if len(food_list) == 0:
        return 0
    if len(food_list) == 1:
        return manhattanDistance(position, food_list[0])

    closest_food = food_list[0]

    for food in food_list:
        if manhattanDistance(position, food) < manhattanDistance(position, closest_food):
            closest_food = food
    return manhattanDistance(position, closest_food)
