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
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Replay memory
from collections import deque

# Neural nets
from DQN import *

params = {
      # Model backups
      'load_file': "saves/model-save_model_21325_125",
      'save_file': "save_model",
      'save_interval': 5000, # original 100000

      # Training parameters
      'train_start': 50,  # Episodes before training starts | orgiginal 5000
      'batch_size': 32,  # Replay memory batch size | original 32
      'mem_size': 100000,  # Replay memory size

      'discount': 0.95,  # Discount rate (gamma value)
      'lr': .002,  # Learning reate
      # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
      # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

      # Epsilon value (epsilon-greedy)
      'eps': 1.0,  # Epsilon start value
      'eps_final': 0.1,  # Epsilon end value
      'eps_step': 10000,  # Epsilon steps between start and end (linear)


    }

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DQN_agent', second = 'DQN_agent', **kwargs):
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
  return [eval(first)(firstIndex, **kwargs), eval(second)(secondIndex, **kwargs)]



##########
# Agents #
##########

class DQN_agent(CaptureAgent):

  """
  DQN agent
  """
  def __init__(self,index, *args, **kwargs):
    CaptureAgent.__init__(self, index)
    # Load parameters from user-given arguments
    self.params = params
    self.params['num_training'] = kwargs.pop('numTraining', 0)

    print("Initialise DQN Agent")

    # Load parameters from user-given arguments
    self.params['width'] = 32 # TODO gameState.data.layout.width
    self.params['height'] = 16 # TODO gameState.data.layout.height

    # Start Tensorflow session
    # TODO Add GPU
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    #self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    self.qnet = DQN(self.params)

    # time started
    self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
    # Q and cost
    self.Q_global = []
    self.cost_disp = 0

    # Stats
    self.cnt = self.qnet.sess.run(self.qnet.global_step)
    self.local_cnt = 0

    self.numeps = 0
    self.last_score = 0
    self.s = time.time()
    self.last_reward = 0.
    self.last_food_difference = 0

    self.replay_mem = deque()
    self.last_scores = deque()
    self.last_food_difference = deque()

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


    #self.start = gameState.getAgentPosition(self.index)
    #print("start ",self.start)
    #state = self.getStateMatrices(gameState)
    #state = self.mergeStateMatrices(state)
    #print(state)

    # Reset reward
    self.last_score = 0
    self.current_score = 0
    self.last_reward = 0.
    self.ep_rew = 0
    self.last_food_difference = 0

    # flag for first state
    self.first_state = True

    # Reset state
    self.last_state = None
    self.current_state = self.getStateMatrices(gameState)

    # Reset actions
    self.last_action = None

    # Reset vars
    self.terminal = None
    self.won = False
    self.Q_global = []
    self.delay = 0

    #reset food status
    self.ourFood = self.CountOurFood(gameState)
    self.theirFood=self.CountTheirFood(gameState)


    # Next
    self.frame = 0
    self.numeps += 1

  def get_direction(self, value):
      if value == 0.:
        return Directions.NORTH
      elif value == 1.:
        return Directions.EAST
      elif value == 2.:
        return Directions.SOUTH
      else:
        return Directions.WEST

  def get_value(self, direction):
    if direction == Directions.NORTH:
      return 0.
    elif direction == Directions.EAST:
      return 1.
    elif direction == Directions.SOUTH:
      return 2.
    else:
      return 3.

  def getMove(self, gameState):
    # Exploit / Explore
    if np.random.rand() > self.params['eps']:
      # Exploit action
      self.Q_pred = self.qnet.sess.run(
        self.qnet.y,
        feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                             (1, self.params['width'], self.params['height'], 7)),
                     self.qnet.q_t: np.zeros(1),
                     self.qnet.actions: np.zeros((1, 4)),
                     self.qnet.terminals: np.zeros(1),
                     self.qnet.rewards: np.zeros(1)})[0]

      self.Q_global.append(max(self.Q_pred))
      a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

      if len(a_winner) > 1:
        move = self.get_direction(
          a_winner[np.random.randint(0, len(a_winner))][0])
      else:
        move = self.get_direction(
          a_winner[0][0])
    else:
      # Random:
      move = self.get_direction(np.random.randint(0, 4))

    # Save last_action
    self.last_action = self.get_value(move)

    return move


  def CountOurFood(self,gameState):
    foodgrid = CaptureAgent.getFood(self,gameState)
    count = foodgrid.count()
    return count

  def CountTheirFood(self, gameState):
    foodgrid = CaptureAgent.getFoodYouAreDefending(self, gameState)
    count = foodgrid.count()
    return count

  def updateLastReward(self,gameState):
    # Process current experience reward
    # TODO CHAANGE REWRDS
    # -1 for loosing time -> nothing happens
    # +1 eat food -> store increases
    # +5 eat Pill -> scares ghosts
    # -100 get eaten by ghost / pacman -> in starting positon and nothing changes
    # +10 positive score / drop
    # +50 eat ghost / eat pacman
    # -2 our food gets eaten

    self.current_score = CaptureAgent.getScore(self,gameState)
    reward = self.current_score - self.last_score

    #self.ourFood = self.ourFood - self.CountOurFood(gameState)
    #self.theirFood = self.theirFood - self.CountTheirFood(gameState)

    #reward = reward + self.ourFood - self.theirFood #adding the more food we have over them

    self.last_score = self.current_score

    if self.first_state: #since we will start in the starting position duh
      self.first_state = False
      #if gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index):
      #  reward -= 20  # we were killed and spawned back to start



    # # todo, blue team has negative reward
    # if reward > 10:
    #   self.last_reward = 50.    # Dropped more than 10 candy in own field
    # elif reward > 0:
    #   self.last_reward = 10.    # Dropped less than 20 candy in own field
    # elif reward < -10:
    #   self.last_reward = -10.  # Get eaten   (Ouch!) -500
    #   self.won = False
    # elif reward <= 0:
    #   self.last_reward = -1.    # Punish time (Pff..)

    if (gameState.isOver()):
      if CaptureAgent.getScore(self,gameState) > 0:
        self.won = True
      if (self.terminal and self.won):
        self.last_reward = 500. #win is great
    else:
      if reward > 10:
        self.last_reward = 50
      elif reward > 0:
        self.last_reward = 10.    # Dropped less than 20 candy in own field
      elif reward < -10:
        self.last_reward = -200.  # Get eaten   (Ouch!) -500
      elif reward == 0:
        self.last_reward = -1



  def observation_step(self, gameState):
    if self.last_action is not None:
      # Process current experience state
      self.last_state = np.copy(self.current_state)
      self.current_state = self.getStateMatrices(gameState)

      self.updateLastReward(gameState) #update reward
      self.ep_rew += self.last_reward

      # Store last experience into memory
      experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
      self.replay_mem.append(experience)
      if len(self.replay_mem) > self.params['mem_size']:
        self.replay_mem.popleft()

      # Save model
      if(params['save_file']):
        if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
          self.qnet.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
          print('Model saved')

      # Train
      self.train()

    # Next
    self.local_cnt += 1
    self.frame += 1
    self.params['eps'] = max(self.params['eps_final'],
                             1.00 - float(self.cnt)/ float(self.params['eps_step']))


  def observationFunction(self, state):
    # Do observation
    self.terminal = False
    self.observation_step(state)

    return state

  def final(self, gameState):
    # Next
    self.ep_rew += self.last_reward

    # Do observation
    self.terminal = True
    self.observation_step(gameState)

    # Print stats
    # Todo: Enable printout
    log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
    log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                   (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
    log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
    sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                     (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
    sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
    sys.stdout.flush()

  def train(self):
    # Train
    if (self.local_cnt > self.params['train_start']):
      batch = random.sample(self.replay_mem, self.params['batch_size'])
      batch_s = [] # States (s)
      batch_r = [] # Rewards (r)
      batch_a = [] # Actions (a)
      batch_n = [] # Next states (s')
      batch_t = [] # Terminal state (t)

      for i in batch:
        batch_s.append(i[0])
        batch_r.append(i[1])
        batch_a.append(i[2])
        batch_n.append(i[3])
        batch_t.append(i[4])
      batch_s = np.array(batch_s)
      batch_r = np.array(batch_r)
      batch_a = self.get_onehot(np.array(batch_a))
      batch_n = np.array(batch_n)
      batch_t = np.array(batch_t)

      self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


  def get_onehot(self, actions):
    """ Create list of vectors with 1 values at index of action in list """
    actions_onehot = np.zeros((self.params['batch_size'], 4))
    for i in range(len(actions)):
      actions_onehot[i][int(actions[i])] = 1
    return actions_onehot

  def chooseAction(self, gameState):
      """
      This will be our main method from where we get the action!
      """
      move = self.getMove(gameState)

      # Stop moving when not legal
      legal = gameState.getLegalActions(self.index)
      if move not in legal:
        move = Directions.STOP

      return move

  """CODE FROM PAPER TO GET STATE SPACE"""

  def mergeStateMatrices(self, stateMatrices):
    """ Merge state matrices to one state tensor """
    stateMatrices = np.swapaxes(stateMatrices, 0, 2)
    total = np.zeros((8, 8))
    for i in range(len(stateMatrices)):
      total += (i + 1) * stateMatrices[i] / 7
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
      capsules = self.getCapsulesYouAreDefending(state)
      matrix = np.zeros((height, width), dtype=np.int8)

      for i in capsules:
        # Insert capsule cells vertically reversed into matrix
        matrix[-1 - i[1], i[0]] = 1

      return matrix

    def GetTheirCapsulesMatrix(state):
      """ Return matrix with capsule coordinates set to 1 """
      width, height = state.data.layout.width, state.data.layout.height
      capsules = self.getCapsules(state)
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



##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def __init__(self, index, *args, **kwargs):
    CaptureAgent.__init__(self, index)

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
