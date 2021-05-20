# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:44:35 2021

@author: 82520
"""
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
from baselineTeam import DefensiveReflexAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################
#StayAgent   DefensiveReflexAgent
def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproxQLearningOffense', second = 'DefensiveReflexAgent', **args):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]



class StayAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
  def chooseAction(self, gameState):
    return 'Stop' 

class ApproxQLearningOffense(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index) #！
    self.startEpisode()
    if self.episodesSoFar == 0:
        print ('We will conduct %d episodes of Training in total' % (self.numTraining))
# learningAgents.py
# -----------------
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
# =============================================================================
  def __init__(self,alpha=99999999, epsilon=99999999, gamma=99999999, numTraining =99999999,**args):
    """
    actionFn: Function which takes a state and returns the list of legal actions

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    #def __init__(self, index, timeForComputing=.1, numTraining=0, epsilon=0.5, alpha=0.1, gamma=1, **args):
    CaptureAgent.__init__(self, 0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! red=0   blue=1
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    
    self.eps_start=0.9
    self.eps_end=0.05 
    self.epsilon =self.eps_start 
    self.alpha =0.2
    self.discount = 0.9
    self.numTraining = 0
    self.qValues = util.Counter()  #init
    #self.index=0
    #reinforcement agent
    #if actionFn == None:
    #    actionFn = lambda state: state.getLegalActions()

    #start training 
    if self.numTraining > 0  and self.episodesSoFar==0:
      print('start training')
      print('epsilon:'+str(self.epsilon)+'alpha:'+str(self.alpha)+'discount:'+str(self.discount))
      self.weights= util.Counter()
    
   #just testing
    if self.numTraining==0:
      self.epsilon = 0.0 # no exploration
      self.alpha = 0.0 # no learning
      
      #our best
      self.weights ={'bias': 455.73482272348315, '#-of-ghosts-1-step-away': -17.618432208826185, 'closest-food': -10.767724871479906, 'eats-food': 202.20764916111045}
  def getPolicy(self, state):
    return self.computeActionFromQValues(state)
  def getValue(self, state):
    return self.computeValueFromQValues(state)
  def getWeights(self):
    return self.weights
  def observeTransition(self, state,action,nextState,deltaReward):
    """
        Called by environment to inform agent that a transition has
        been observed. This will result in a call to self.update
        on the same arguments

        NOTE: Do *not* override or call this function
    """
    self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward)
 
  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0    

  def stopEpisode(self):
    """
      Called by environment when episode is done
    """
    print('episodesSoFar'+str(self.episodesSoFar))
    if self.episodesSoFar < self.numTraining:
        self.accumTrainRewards += self.episodeRewards
    else:
        self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
        # Take off the training wheels
        self.epsilon = 0.0    
        self.alpha = 0.0
    else:
        # # decrease epsilon
        self.epsilon = max(self.eps_end,self.eps_start-(self.eps_start-self.eps_end)* self.episodesSoFar/(self.numTraining*0.9))
        print('epsilon:'+str(self.epsilon)+'alpha:'+str(self.alpha)+'discount:'+str(self.discount))
  def isInTraining(self):
    return self.episodesSoFar < self.numTraining

  def isInTesting(self):
    return not self.isInTraining()

  ################################
  # Controls needed for Crawler  #
  ################################
# =============================================================================
#   def setEpsilon(self, epsilon):
#     self.epsilon = epsilon
# 
#   def setLearningRate(self, alpha):
#     self.alpha = alpha
# 
#   def setDiscount(self, discount):
#     self.discount = discount
# =============================================================================
  def doAction(self,state,action):
    """
        Called by inherited class when
        an action is taken in a state
    """
    self.lastState = state
    self.lastAction = action

  ###################
  # Pacman Specific #
  ###################
  def observationFunction(self, state):
    """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
    """
    if not self.lastState is None:
        reward = self.calculateReward(self.lastState,state)
        self.observeTransition(self.lastState, self.lastAction, state, reward)
    return state

  def final(self, state):
    """
      Called by Pacman game at the terminal state
    """
    CaptureAgent.final(self, state) #let self.observationHistory = []
    reward = self.calculateTerminalReward(self.lastState, state)
    self.observeTransition(self.lastState, self.lastAction, state, reward)
    self.stopEpisode()

    # Make sure we have this var
    if not 'episodeStartTime' in self.__dict__:
        self.episodeStartTime = time.time()
    if not 'lastWindowAccumRewards' in self.__dict__:
        self.lastWindowAccumRewards = 0.0
    self.lastWindowAccumRewards += state.getScore()

    NUM_EPS_UPDATE =5
    if self.episodesSoFar % NUM_EPS_UPDATE == 0:
        print('epsilon:'+str(self.epsilon)+'alpha:'+str(self.alpha)+'discount:'+str(self.discount))
        print ('Reinforcement Learning Status:')
        windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
        if self.episodesSoFar <= self.numTraining:
            trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
            print ('\tCompleted %d out of %d training episodes' % (
                   self.episodesSoFar,self.numTraining))
            print ('\tAverage Rewards over all training: %.2f' % (
                    trainAvg))
        else:
            testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
            print ('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
            print ('\tAverage Rewards over testing: %.2f' % testAvg)
        print ('\tAverage Rewards for last %d episodes: %.2f'  % (
                NUM_EPS_UPDATE,windowAvg))
        print ('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
        self.lastWindowAccumRewards = 0.0
        self.episodeStartTime = time.time()

        print('weights is')
        print(self.weights)
        
    if self.episodesSoFar == self.numTraining:
        msg = 'Training Done (turning off epsilon and alpha)'
        print ('%s\n%s' % (msg,'-' * len(msg)))


    if self.episodesSoFar == self.numTraining:
        #self.saveWeights(filename)
        CaptureAgent.final(self, state)
        print('weights is')
        print(self.weights)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    #from beselineteam.py
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

#  given the weight, the following code is used to test (remember setting numTraining==0 )
  def chooseAction(self, gameState):
    
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    
    #from baselineTeam.py
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      self.doAction(gameState, bestAction)
      return bestAction
  
    else:
# =============================================================================
#         # You can profile your evaluation time by uncommenting these lines
#         # start = time.time()
#         values = [self.evaluate(gameState, a) for a in actions]
#         # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
#         maxValue = max(values)
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]
#         return random.choice(bestActions)
# =============================================================================
        if util.flipCoin(self.epsilon):
          bestAction = random.choice(actions) 
        else:
          bestAction = self.getPolicy(gameState)
        self.doAction(gameState, bestAction)
        return bestAction

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    return self.getFeatures(state, action) * self.weights

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
      
    """
    "*** YOUR CODE HERE ***"
    #from baselineTeam.py
    actions = gameState.getLegalActions(self.index)
    values = [self.getQValue(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

# for training
  def computeValueFromQValues(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    return self.getQValue(state, self.getPolicy(state))


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    target=(reward + self.discount * self.getValue(nextState)) 
    diff = target - self.getQValue(state, action)
    for feature,featurevalue in self.getFeatures(state, action).items():
        self.weights[feature] += self.alpha * diff *featurevalue


  def calculateReward(self,lastState, state):

    #food
    currentfoods = self.getFood(lastState).asList()
    nextFoods = self.getFood(state).asList()
    if len(currentfoods) - len(nextFoods) == 1:
      reward = 10       
      return reward
     
    if state.getScore() >self.lastState.getScore():
        reward=(state.getScore()-self.lastState.getScore())*100
        return reward
    
    #kill
    for ghost in self.getOpponents(state):
        if state.getAgentState(ghost).getPosition()==state.getAgentState(self.index).getPosition():
            reward= -100
            return reward

    reward=0
    return reward
    

  def calculateTerminalReward(self,lastState, state):
    return  state.getScore() - self.lastState.getScore() 
 
# # feature
# featureExtractors.py
# --------------------
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
  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = self.getFood(state)
    walls = state.getWalls()
    
    ghosts=[]
    for i in self.getOpponents(state):
        positions=state.getAgentState(i).getPosition()
        if positions!=None:
            ghosts.append(positions)

    features = util.Counter()
    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    x, y = state.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)    
    
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        features["eats-food"] = 1.0
    dist = self.closestFood((next_x, next_y), food, walls)
    if dist is not None:
        # make the distance a number less than one otherwise the update
        # will diverge wildly
        features["closest-food"] = float(dist) / (walls.width * walls.height)
    features.divideAll(10.0)
    return features
  def closestFood(self,pos, food, walls):
    """
    queue bfs
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
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None