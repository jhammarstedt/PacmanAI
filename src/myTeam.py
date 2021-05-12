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
from game import Directions
import game
import numpy as np

params = {
      # Model backups
      'load_file': None,
      'save_file': None,
      'save_interval': 10000,

      # Training parameters
      'train_start': 5000,  # Episodes before training starts
      'batch_size': 32,  # Replay memory batch size
      'mem_size': 100000,  # Replay memory size

      'discount': 0.95,  # Discount rate (gamma value)
      'lr': .0002,  # Learning reate
      # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
      # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

      # Epsilon value (epsilon-greedy)
      'eps': 1.0,  # Epsilon start value
      'eps_final': 0.1,  # Epsilon end value
      'eps_step': 10000  # Epsilon steps between start and end (linear)
    }

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DQN_agent', second = 'DummyAgent'):
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
  #return [eval(DQN_agent),eval(DQN_agent))]  # maybe like this
  print(f"FIRST TEAM {first}")
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


##########
# Agents #
##########

class DQN_agent(CaptureAgent):

  # def __init__(self, args):
  #    super().__init__()
  #
  #   print("Initialise DQN Agent")
  #
  #   # Load parameters from user-given arguments
  #   #
  #   # self.params['width'] = args['width']
  #   # self.params['height'] = args['height']
  #   # self.params['num_training'] = args['numTraining']

  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''

    self.params = params

    self.params['width'] = gameState.data.layout.width
    self.params['height'] = gameState.data.layout.height



    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    print("start ",self.start)
    state = self.getStateMatrices(gameState)
    state = self.mergeStateMatrices(state)
    print(state)

    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    This will be our main method from where we get the action!
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

  """CODE FROM PAPER TO GET STATE SPACE"""

  def mergeStateMatrices(self, stateMatrices):
    """ Merge state matrices to one state tensor """
    stateMatrices = np.swapaxes(stateMatrices, 0, 2)
    total = np.zeros((7, 7))
    for i in range(len(stateMatrices)):
      total += (i + 1) * stateMatrices[i] / 6
    return total

  def getStateMatrices(self, state):

    """ Return wall, ghosts, food, capsules matrices """



    def getWallMatrix(state):
      """ Return matrix with wall coordinates set to 1 """
      width, height = state.data.layout.width, state.data.layout.height
      grid = state.data.layout.walls
      matrix = np.zeros((height, width), dtype=np.int8)
      for i in range(grid.height):
        for j in range(grid.width):
          # Put cell vertically reversed in matrix
          cell = 1 if grid[j][i] else 0
          matrix[-1 - i][j] = cell
      return matrix

    def getFriendPacmanMatrix(state):
      """ Return matrix with pacman coordinates set to 1 """
      width, height = state.data.layout.width, state.data.layout.height
      matrix = np.zeros((height, width), dtype=np.int8)
      team = self.getTeam(state)

      for agent in team:
        pos = state.getAgentPosition(agent)
        # TODO distinguish between pacman and ghost
        cell = 1
        matrix[-1 - int(pos[1])][int(pos[0])] = cell

      return matrix

    def getEnemyPacmanMatrix(state):
      """ Return matrix with ghost coordinates set to 1 """
      width, height = state.data.layout.width, state.data.layout.height
      matrix = np.zeros((height, width), dtype=np.int8)
      enemies = self.getOpponents(state)
      
      for agent in enemies:
        # TODO use probabilities if the
        pos = state.getAgentPosition(agent)
        if pos is not None:
          cell = 1
          matrix[-1 - int(pos[1])][int(pos[0])] = cell

      # OLD CODE
      #for agentState in state.data.agentStates:
       # if not agentState.isPacman:
       #   if not agentState.scaredTimer > 0:
       #     pos = agentState.configuration.getPosition()
       #     cell = 1
       #     matrix[-1 - int(pos[1])][int(pos[0])] = cell

      return matrix

    def GetOurFoodMatrix(state):
      width, height = state.data.layout.width, state.data.layout.height
      grid = self.getFoodYouAreDefending(state)
      matrix = np.zeros((height,width), dtype = np.int8)

      for i in range(grid.height):
        for j in range(grid.width):
          # Put cell vertically reversed in matrix
          cell = 1 if grid[j][i] else 0
          matrix[-1 - i][j] = cell

      return matrix

    def GetTheirFoodMatrix(state):
      width, height = state.data.layout.width, state.data.layout.height
      grid = self.getFood(state)
      matrix = np.zeros((height, width), dtype=np.int8)

      for i in range(grid.height):
        for j in range(grid.width):
          # Put cell vertically reversed in matrix
          cell = 1 if grid[j][i] else 0
          matrix[-1 - i][j] = cell

      return matrix

    def GetOurCapsulesMatrix(state):
      """ Return matrix with capsule coordinates set to 1 """
      width, height = state.data.layout.width, state.data.layout.height
      capsules = self.getCapsulesYouAreDefending()
      matrix = np.zeros((height, width), dtype=np.int8)

      for i in capsules:
        # Insert capsule cells vertically reversed into matrix
        matrix[-1 - i[1], i[0]] = 1

      return matrix

    def GetTheirCapsulesMatrix(state):
      """ Return matrix with capsule coordinates set to 1 """
      width, height = state.data.layout.width, state.data.layout.height
      capsules = self.getCapsules()
      matrix = np.zeros((height, width), dtype=np.int8)

      for i in capsules:
        # Insert capsule cells vertically reversed into matrix
        matrix[-1 - i[1], i[0]] = 1

      return matrix

    # Create observation matrix as a combination of
    # wall, pacman, ghost, food and capsule matrices
    # width, height = state.data.layout.width, state.data.layout.height
    width, height = self.params['width'], self.params['height']
    observation = np.zeros((7, height, width))

    observation[0] = getWallMatrix(state)
    observation[1] = getFriendPacmanMatrix(state) #our
    observation[2] = getEnemyPacmanMatrix(state) #their
    observation[3] = GetOurFoodMatrix(state)
    observation[4] = GetTheirFoodMatrix(state)
    observation[5] = GetOurCapsulesMatrix(state)
    observation[6] = GetTheirCapsulesMatrix(state)

    """
    We need 
    Opponent ghosts
    Our ghosts
      
    -- Getourplayer
    -- GetTheirplayer
    -- GetOurFood
    -- GetTheirFood
    
    -- maybe: Get Our and their ScaredGhost,Ghost, Capsule 
    
    """



    observation = np.swapaxes(observation, 0, 2)

    return observation



