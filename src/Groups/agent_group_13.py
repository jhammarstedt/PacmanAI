# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint
import json

#################
# Team creation #
#################

NUM_TRAINING = 0
NUM_GAMES = 0
TRAINING = False

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproxQLearningOffense', second = 'ApproxQLearningDefense', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  #NUM_TRAINING = numTraining
  #NUM_GAMES = numGames
  #TRAINING = training_indicator
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


class ApproxQLearningOffense(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.epsilon = 0.1
    self.alpha = 0.2
    self.discount = 0.9
    #NUM_TRAINING = self.training_eps
    #TRAINING = self.training
    self.episodesSoFar = 0
    #For competition only:
    #"""
    self.training = False
    self.incoming_weights = 0
    self.incoming_weights_2 = 0
    self.training_eps = 0
    #"""
    #print("Training??")
    #print(self.training)
    #print("what episode?")
    #print(self.training_eps)
    #print("Input weights? (Offense)")
    #print(self.incoming_weights)
    #print(self.episodesSoFar)
    if not self.training:
        #OUR PRE-FINAL POLICY
                        
        #our weights after x eps with offense feature group           
        self.weights = {'closest-food': -862.9138960460731, 
                        'bias': 151.20286179765637, 
                        '#-of-ghosts-1-step-away': -53.65355688594745, 
                        '#-of-ghosts-2-steps-away': -99.27385741803461, 
                        '#-of-scared-ghosts-1-step-away': 0.3459887911549833, 
                        '#-of-scared-ghosts-2-steps-away': 2.355173692883608, 
                        'ghost-danger': 53.46748024408441, 
                        'eats-food': 127.62985520783778}
    else:
        self.weights = self.incoming_weights
    
    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
        Picks among the actions with the highest Q(s,a).
    """
    
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in legalActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = None
    #print("are we still training nao?")
    #print(self.training)
    if self.training:
      #print("scooch amooch")
      for action in legalActions:
        self.updateWeights(gameState, action)
    if not util.flipCoin(self.epsilon) or not self.training:
      # exploit
      #print("WE AINT TRAINING, SHALL EXPLOIT AND WITH THEEESE: ")
      #print(self.weights)
      action = self.getPolicy(gameState)
    else:
      # explore
      #print("EXPLORE LIKE A PIONEER")
      action = random.choice(legalActions)
    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # features vector
    features = self.featuresExtractor.getFeatures(gameState, action)
    Q = 0
    for feature in self.weights.keys():
      Q += self.weights[feature] * features[feature]
    return Q

  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featuresExtractor.getFeatures(gameState, action)
    oldValue = self.getQValue(gameState, action)
    futureQValue = self.getValue(nextState)
    difference = (reward + self.discount * futureQValue) - oldValue
    #print("reward: " + str(reward))
    #print("futureQ: " + str(futureQValue))
    #print("oldVal: " + str(oldValue))
    #print(difference)
    # for each feature i
    for feature in self.weights.keys():
      newWeight = self.alpha * difference * features[feature]
      self.weights[feature] += newWeight
    #print("NEWLY UPDATED WEIGHTS OFFENSE")
    #print(self.weights)
    #print("offensive keys!")
    #print(self.weights.keys())
    

  def updateWeights(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    reward = self.getReward(gameState, nextState)
    self.update(gameState, action, nextState, reward)

  def getReward(self, gameState, nextState):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)
    enemiesPos = [gameState.getAgentPosition(ene) for ene in self.getOpponents(gameState)]
    #print("here art I: ")
    #print(agentPosition)
    #print("there theys at: ")
    #print(enemiesPos)
    # check if I have updated the score
    #features to represent our team
    #pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    
    
    #symmetrical features for the opponents    
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward += diff * 10

    # check if food eaten in nextState
    myFoods = self.getFood(gameState).asList()
    distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextFoods = self.getFood(nextState).asList()
      if len(myFoods) - len(nextFoods) == 1:
        reward += 10
        

    # check if I am eaten
    #enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    #ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward += -100
          
          
          
    # check if we eat enemy
    if len(scared_ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in scared_ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I kill in the next state
          reward += 50
          
      
    return reward

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    #print("weights in the end (offense)")
    #print(self.weights)
    self.episodesSoFar += 1
    self.incoming_weights = self.weights
    #if self.training_eps%50 == 0:
    #    f = open("weights_offense.txt", "a")
    #    f.write("Episode " + str(self.training_eps) + "\n")
    #    weight_string = json.dumps(self.weights)
    #    f.write(weight_string + "\n")
    #    f.close()
    # did we finish training?

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    return self.getQValue(gameState, bestAction)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    actionVals = {}
    bestQValue = float('-inf')
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [k for k, v in actionVals.items() if v == bestQValue]
    # random tie-breaking
    return random.choice(bestActions)

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)

  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)


class ApproxQLearningDefense(CaptureAgent):
  def registerInitialState(self, gameState):
    self.epsilon = 0.1
    self.alpha = 0.2
    self.discount = 0.9
    self.start = gameState.getAgentPosition(self.index)
    #NUM_TRAINING = self.training_eps
    #TRAINING = self.training
    self.episodesSoFar = 0
    #For competition only:
    #"""
    self.training = False
    self.incoming_weights = 0
    self.incoming_weights_2 = 0
    self.training_eps = 0
    #"""
    
    #print("Training??")
    #print(self.training)
    #print("what episode?")
    #print(self.training_eps)
    #print("Input weights? (Defense)")
    #print(self.incoming_weights_2)
    #print(self.episodesSoFar)
    if not self.training:
        #OUR PREFINAL WEIGHTS
        #our weights after X eps with defensive feature group
        """
        self.weights = {"closest-prey": 0.007146985500778681, 
                        "bias": -0.3338519152700928, 
                        "#-of-invaders-1-step-away": -1.3270266497784708, 
                        "#-of-invaders-2-steps-away": -5.038870308964941, 
                        "#-of-edible-invaders-1-step-away": 0.0, 
                        "#-of-edible-invaders-2-steps-away": 0.0, 
                        "dinner-served": 0.0, 
                        "enemy-eats-food": 0.00042177103868804293}
        """              
        self.weights = {'closest-prey': -200.9138960460731, 
                        'bias': 4.794054831046247, 
                        '#-of-invaders-1-step-away': -59.8823974479147, 
                        '#-of-invaders-2-steps-away': -8.436660685889889, 
                        '#-of-edible-invaders-1-step-away': 2.2258945447236966, 
                        '#-of-edible-invaders-2-steps-away': 20.010512578649813, 
                        'dinner-served': -1.778943991005062, 
                        'enemy-eats-food': -59.118848739157094}



        
    else:
        self.weights = self.incoming_weights_2
    
    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
        Picks among the actions with the highest Q(s,a).
    """
    
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in legalActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = None
    #print("are we still training nao?")
    #print(self.training)
    if self.training:
      #print("scooch amooch")
      for action in legalActions:
        self.updateWeights(gameState, action)
    if not util.flipCoin(self.epsilon) or not self.training:
      # exploit
      #print("WE AINT TRAINING, SHALL EXPLOIT AND WITH THEEESE: ")
      #print(self.weights)
      action = self.getPolicy(gameState)
    else:
      # explore
      #print("EXPLORE LIKE A PIONEER")
      action = random.choice(legalActions)
    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # features vector
    features = self.featuresExtractor.getFeatures(gameState, action)
    Q = 0
    for feature in self.weights.keys():
      Q += self.weights[feature] * features[feature]
    return Q

  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featuresExtractor.getFeatures(gameState, action)
    oldValue = self.getQValue(gameState, action)
    futureQValue = self.getValue(nextState)
    difference = (reward + self.discount * futureQValue) - oldValue
    #print("reward: " + str(reward))
    #print("futureQ: " + str(futureQValue))
    #print("oldVal: " + str(oldValue))
    #print(difference)
    # for each feature i
    for feature in self.weights.keys():
      newWeight = self.alpha * difference * features[feature]
      self.weights[feature] += newWeight
    #print("NEWLY UPDATED WEIGHTS DEFENSE")
    #print(self.weights)
    #print("defensive keys!")
    #print(self.weights.keys())

  def updateWeights(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    reward = self.getReward(gameState, nextState)
    self.update(gameState, action, nextState, reward)

  def getReward(self, gameState, nextState):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)
    enemiesPos = [gameState.getAgentPosition(ene) for ene in self.getOpponents(gameState)]
    #print("here art I: ")
    #print(agentPosition)
    #print("there theys at: ")
    #print(enemiesPos)
    # check if I have updated the score
    #features to represent our team
    #pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    
    
    #symmetrical features for the opponents
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer > 0]
    edible_invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer <= 0]
    
    
    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward += diff * 10

    # check GUARDED food gets eaten in nextState
    myStash = self.getFoodYouAreDefending(gameState).asList()
    if sum([x != None for x in enemiesPos ]) > 0:
        distToFood = min([self.getMazeDistance(enemyPos, food) for food in myStash for enemyPos in enemiesPos if enemyPos != None])
    # I they are 1 step away, will they be able to eat it?
        if distToFood == 1:
          nextStash = self.getFoodYouAreDefending(nextState).asList()
          if len(myStash) - len(nextStash) == 1:
            reward += -10

    # check if I am eaten
    #enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    #ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
          
    if len(invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward += -50
          
    if len(edible_invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in edible_invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I kill in the next state
          reward += 100
      
    return reward

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    #print("weights in the end (defense)")
    #print(self.weights)
    self.episodesSoFar += 1
    self.incoming_weights_2 = self.weights
    #if self.training_eps%50 == 0:
        #f = open("weights_defense.txt", "a")
        #f.write("Episode " + str(self.training_eps) + "\n")
        #weight_string = json.dumps(self.weights)
        #f.write(weight_string + "\n")
        #f.close()
    # did we finish training?

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    return self.getQValue(gameState, bestAction)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    actionVals = {}
    bestQValue = float('-inf')
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [k for k, v in actionVals.items() if v == bestQValue]
    # random tie-breaking
    return random.choice(bestActions)

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)

  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)
    
    
class FeaturesExtractor:

  def __init__(self, agentInstance):
    self.agentInstance = agentInstance

  def getFeatures(self, gameState, action):
    #FOLLOWING Gnanasekaran, Feliu Faba and An we use:
    # --> ghosts/scared ghosts 1 and 2 steps away
    # --> binary: ghost prescence 1 step away
    # -->  "eating food"
    # --> distance to closest food
    #the code is still based on Sharma's implementation
    
    #we also add:
    # --> stash (food we guard)
    # --> attacking, defending, defending and scared teammates
    # extract the grid of food and wall locations and get the ghost locations
    me = gameState.getAgentState(self.agentInstance.index)
    bag = me.numCarrying
    home = self.agentInstance.start
    food = self.agentInstance.getFood(gameState)
    stash = self.agentInstance.getFoodYouAreDefending(gameState)
    walls = gameState.getWalls()
    caps = self.agentInstance.getFoodYouAreDefending(gameState)
    
    pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.agentInstance.getOpponents(gameState)]
    
    #features to represent our team
    attackers = [a.getPosition() for a in pals if a.isPacman and a.getPosition() != None]
    defenders = [a.getPosition() for a in pals if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_defenders = [a.getPosition() for a in pals if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    #symmetrical features for the opponents
    invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.agentInstance.index).scaredTimer > 0]
    edible_invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.agentInstance.index).scaredTimer <= 0]
    
    ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    
    
    # ghosts = state.getGhostPositions()

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    agentPosition = gameState.getAgentPosition(self.agentInstance.index)
    x, y = agentPosition
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in ghosts]
    features["#-of-ghosts-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #ADD ALSO GHOSTS 2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-ghosts-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    
    #Do the same for scared ghosts
    #1 STEP AWAY:
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in scared_ghosts]
    features["#-of-scared-ghosts-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-scared-ghosts-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    
    #"""
    #count (creepy) invaders 1-step away
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in invaders]
    features["#-of-invaders-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #ADD ALSO INVADERS 2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-invaders-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    
    #The same for edible invaders:
    #1 STEP AWAY
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in edible_invaders]
    features["#-of-edible-invaders-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-edible-invaders-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    #"""

        
    #BINARY: GHOST/EDIBLE PACMAN 1 STEP AWAY
    features["ghost-danger"] = 1.0*(features["#-of-ghosts-1-step-away"] > 0)
    features["dinner-served"] = 1.0*(features["#-of-edible-invaders-1-step-away"] > 0)
    
    
    # if len(ghosts) > 0:
    #   minGhostDistance = min([self.agentInstance.getMazeDistance(agentPosition, g) for g in ghosts])
    #   if minGhostDistance < 3:
    #     features["minGhostDistance"] = minGhostDistance

    # successor = self.agentInstance.getSuccessor(gameState, action)
    # features['successorScore'] = self.agentInstance.getScore(successor)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist = self.closestFood((next_x, next_y), food, walls)
    dist_2 = self.furthestFood((next_x, next_y), caps, walls)
    dist_home = self.getMazeDistancePro((next_x, next_y), home, walls)
    #print("to food!")
    #print(dist)
    #print("to caps")
    #print(dist_2)
    #print("distance home!" + str(dist_home))
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      if bag < 5:
        features["closest-food"] = float(dist) / (walls.width * walls.height)
      else:
        features["closest-food"] = float(dist_home) / (walls.width * walls.height)
       
    if dist_2 is not None:
        distToPrey = dist_2
    else:
        distToPrey = None
    myStash = stash.asList()
    features["enemy-eats-food"] = 0.0
    distToFood = 1000
    if sum([x != None for x in edible_invaders]) > 0:
        #print("PAY ATTENTION THE TEST IS HERE")
        if len(myStash) > 0:
            distToFood = min([self.agentInstance.getMazeDistance(enemyPos, food) for food in myStash for enemyPos in edible_invaders])
        distToPrey = min([self.getMazeDistancePro((next_x, next_y), enemyPos, walls) for enemyPos in edible_invaders])
        # I they are 1 step away, will they be able to eat it?
        if distToFood == 1 and not features["#-of-edible-invaders-1-step-away"]:
            features["enemy-eats-food"] = 1.0
    if distToPrey is not None:
        features["closest-prey"] = distToPrey
    # capsules = self.agentInstance.getCapsules(gameState)
    # if len(capsules) > 0:
    #   closestCap = min([self.agentInstance.getMazeDistance(agentPosition, cap) for cap in self.agentInstance.getCapsules(gameState)])
    #   features["closestCapsule"] = closestCap

    
    features.divideAll(10.0)
    #print(features)
    return features

  def closestFood(self, pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None
    
  def furthestFood(self, pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    furthest = None
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        furthest = dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return furthest
    
  def getMazeDistancePro(self, pos, goal, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if goal == (pos_x, pos_y):
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None
    


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
    
class ApproxQLearningFull(CaptureAgent):

  def registerInitialState(self, gameState):
    self.epsilon = 0.1
    self.alpha = 0.2
    self.discount = 0.9
    #NUM_TRAINING = self.training_eps
    #TRAINING = self.training
    self.episodesSoFar = 0
    
    #Sharma's final weights
    print("Training??")
    print(self.training)
    print("what episode?")
    print(self.training_eps)
    print("Input weights?")
    print(self.incoming_weights)
    #print(self.episodesSoFar)
    if not self.training:
        #Sharma's final values
        """
        self.weights = {'closest-food': -3.099192562140742,
                        'bias': -9.280875042529367,
                        '#-of-ghosts-1-step-away': -16.6612110039328,
                        'eats-food': 11.127808437648863}
        
        #our weights after 2579 eps with feature group A
        
        self.weights = {'closest-food': -856.1786744165045, 
                    'bias': 106.68036381557351, 
                    '#-of-ghosts-1-step-away': 12.18079547870186, 
                    '#-of-ghosts-2-steps-away': -109.98449245556566, 
                    '#-of-scared-ghosts-1-step-away': -5.556341920707556, 
                    '#-of-scared-ghosts-2-steps-away': -3.704544509847187, 
                    'ghost-danger': 1.5515033849788007, 
                    'eats-food': 118.4663659006324}
        """
        #our weights after 1850 eps with feature group B (full fledged)
        self.weights = {"closest-food": -390.21416161913766, 
                        "bias": 79.47803612724823, 
                        "#-of-ghosts-1-step-away": 9.587122344866998, 
                        "#-of-ghosts-2-steps-away": -107.80363688489174, 
                        "#-of-scared-ghosts-1-step-away": -7.4024844801450085, 
                        "#-of-scared-ghosts-2-steps-away": 3.0439312062917274, 
                        "#-of-invaders-1-step-away": -1.8547322049767951, 
                        "#-of-invaders-2-steps-away": -4.374498166895807, 
                        "#-of-edible-invaders-1-step-away": -1.6505852213735746, 
                        "#-of-edible-invaders-2-steps-away": -2.0912394562026417, 
                        "dinner-served": -1.6505852213735746, 
                        "ghost-danger": 9.535985514512213, 
                        "eats-food": 113.87407979357295}
                        
        
    else:
        self.weights = self.incoming_weights
    
    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
        Picks among the actions with the highest Q(s,a).
    """
    
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in legalActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = None
    #print("are we still training nao?")
    #print(self.training)
    if self.training:
      #print("scooch amooch")
      for action in legalActions:
        self.updateWeights(gameState, action)
    if not util.flipCoin(self.epsilon) or not self.training:
      # exploit
      #print("WE AINT TRAINING, SHALL EXPLOIT AND WITH THEEESE: ")
      #print(self.weights)
      action = self.getPolicy(gameState)
    else:
      # explore
      #print("EXPLORE LIKE A PIONEER")
      action = random.choice(legalActions)
    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # features vector
    features = self.featuresExtractor.getFeatures(gameState, action)
    return features * self.weights

  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featuresExtractor.getFeatures(gameState, action)
    oldValue = self.getQValue(gameState, action)
    futureQValue = self.getValue(nextState)
    difference = (reward + self.discount * futureQValue) - oldValue
    #print("reward: " + str(reward))
    #print("futureQ: " + str(futureQValue))
    #print("oldVal: " + str(oldValue))
    #print(difference)
    # for each feature i
    for feature in features:
      newWeight = self.alpha * difference * features[feature]
      self.weights[feature] += newWeight
    #print("NEWLY UPDATED WEIGHTS YEEES")
    #print(self.weights)

  def updateWeights(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    reward = self.getReward(gameState, nextState)
    self.update(gameState, action, nextState, reward)

  def getReward(self, gameState, nextState):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)
    enemiesPos = [gameState.getAgentPosition(ene) for ene in self.getOpponents(gameState)]
    #print("here art I: ")
    #print(agentPosition)
    #print("there theys at: ")
    #print(enemiesPos)
    # check if I have updated the score
    #features to represent our team
    #pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    
    
    #symmetrical features for the opponents
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer > 0]
    edible_invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer <= 0]
    
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward += diff * 10

    # check if food eaten in nextState
    myFoods = self.getFood(gameState).asList()
    distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextFoods = self.getFood(nextState).asList()
      if len(myFoods) - len(nextFoods) == 1:
        reward += 10
        
    # check GUARDED food gets eaten in nextState
    myStash = self.getFoodYouAreDefending(gameState).asList()
    if sum([x != None for x in enemiesPos ]) > 0:
        distToFood = min([self.getMazeDistance(enemyPos, food) for food in myStash for enemyPos in enemiesPos if enemyPos != None])
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextStash = self.getFoodYouAreDefending(nextState).asList()
      if len(myStash) - len(nextStash) == 1:
        reward += -10

    # check if I am eaten
    #enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    #ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward += -100
          
    if len(invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward += -50
          
          
    # check if we eat enemy
    if len(scared_ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in scared_ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I kill in the next state
          reward += 50
          
    if len(edible_invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in edible_invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I kill in the next state
          reward += 100
      
    return reward

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    print("weights in the end")
    print(self.weights)
    self.episodesSoFar += 1
    self.incoming_weights = self.weights
    if self.training_eps%50 == 0:
        f = open("weights.txt", "a")
        f.write("Episode " + str(self.training_eps) + "\n")
        weight_string = json.dumps(self.weights)
        f.write(weight_string + "\n")
        f.close()
    # did we finish training?

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    return self.getQValue(gameState, bestAction)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    actionVals = {}
    bestQValue = float('-inf')
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [k for k, v in actionVals.items() if v == bestQValue]
    # random tie-breaking
    return random.choice(bestActions)

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)

  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)