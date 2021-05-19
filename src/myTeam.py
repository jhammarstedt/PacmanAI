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
import tensorflow.compat.v1 as tf
import game
import numpy as np
import sys
import os
from util import nearestPoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Replay memory
from collections import deque

# Neural nets
from DQN import *
from game import Actions
from baselineTeam import ReflexCaptureAgent,DefensiveReflexAgent

load_model = False
if load_model:
    with open("saves/checkpoint") as f:
        data = f.readline()
        f.close()
    load = data[24:-2]  # quick fix to read the model from checkpoint
    load = f"saves/{load}"
    print(load)
else:
    load = None


params = {
    # Model backups
    'load_file': load,
    'save_file': "v1",
    'save_interval': 55000,  # original 100000

    # Training parameters
    'TRAIN': True,
    'train_start': 5000,  # Episodes before training starts | orgiginal 5000
    'batch_size': 32,  # Replay memory batch size | original 32
    'mem_size': 100000,  # Replay memory size

    'discount': 0.95,  # Discount rate (gamma value)
    'lr': .0002,  # Learning reate

    # Epsilon value (epsilon-greedy)
    'eps': 0.3,  # Epsilon start value
    'eps_final': 0.3,  # Epsilon end value
    'eps_step': 100000,  # Epsilon steps between start and end (linear)

    # State matrices
    'STATE_MATRICES': 14,

    # Enable GPU
    'GPU': False

}


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='terminator', second='OffensiveAgent', **kwargs):
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
    # return [eval(DQN_agent),eval(DQN_agent))]  # maybe like this
    print(f"PLayer 1: {first} red")
    print(f"Player 2: {second} orange")
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex, **kwargs), eval(second)(secondIndex, **kwargs)]


##########
# Agents #
##########

class DQN_agent(CaptureAgent):
    """
  DQN agent
  """

    def __init__(self, index, *args, **kwargs):
        CaptureAgent.__init__(self, index)
        # Load parameters from user-given arguments
        self.params = params
        self.params['num_training'] = kwargs.pop('numTraining', 0)



        # Load parameters from user-given arguments
        self.params['width'] = 34  # TODO gameState.data.layout.width
        self.params['height'] = 18  # TODO gameState.data.layout.height
        if self.params['TRAIN']:
            self.params['eps'] = 0.
            params['eps'] =0.
        # Start Tensorflow session
        # TODO Add GPU
        if self.params['GPU']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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

        # reset food status
        self.ourFood = self.CountOurFood(gameState)
        self.theirFood = self.CountTheirFood(gameState)

        # first Astar actions
        self.atCenter = False  # first walk to center before we start DQN
        self.center_counter = 0  # the list we will go through to get

        center_point = self.getCenterPos(gameState)
        self.ASTARPATH = self.getCenterPos(gameState)
        # Next
        self.frame = 0
        self.numeps += 1

    def isWall(self,gameState,pos:tuple):
        grid = gameState.data.layout.walls
        return grid[pos[0]][pos[1]]

    def getCenterPos(self,gameState):
        width = self.params['width']
        height = self.params['height']
        # ASTAR Path to center

        if gameState.isOnRedTeam(self.index):
            pos_x = int(width / 2) - 1
            for i in range(1000):
              pos_y = random.randint(int(height / 4), int(0.75 * height))

              center = (pos_x,pos_y)
              if not self.isWall(gameState,center):
                  return deque(self.aStarSearch(gameState.getAgentPosition(self.index), gameState,
                                                [center]))  # hard code for now
        else: #blue
            pos_x = int(width / 2) + 1
            for i in range(1000):
                pos_y = random.randint(int(height / 4), int(0.75 * height))
                center = (pos_x, pos_y)
                if not self.isWall(gameState, center):
                    return deque(self.aStarSearch(gameState.getAgentPosition(self.index), gameState,
                                                  [center]))  # hard code for now

        #center_red = [(16, 10), (16, 7)]
        #center_blue = [(17, 7), (17, 10)] #! needs to be fixed
        #i = random.randint(0,1)



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
        if np.random.rand() > self.params['eps'] or not self.params['TRAIN']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict={self.qnet.x: np.reshape(self.current_state,
                                                   (1, self.params['width'], self.params['height'],
                                                    self.params['STATE_MATRICES'])),
                           self.qnet.q_t: np.zeros(1),
                           self.qnet.actions: np.zeros((1, 4)),
                           self.qnet.terminals: np.zeros(1),
                           self.qnet.rewards: np.zeros(1)})[0]


            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
                print("Selected move: "+str(move))
            else:
                move = self.get_direction(
                    a_winner[0][0])
                print("Q_pred: "+str(self.Q_pred)+" Selected move: "+str(move))
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))
            print("Random move: "+str(move))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def CountOurFood(self, gameState):
        foodgrid = CaptureAgent.getFood(self, gameState)
        count = foodgrid.count()
        return count

    def CountTheirFood(self, gameState):
        foodgrid = CaptureAgent.getFoodYouAreDefending(self, gameState)
        count = foodgrid.count()
        return count

    def updateLastReward(self,currentGameState):

        # GameState objects
        lastGameState = self.getCurrentObservation()
        #currentGameState = self.getCurrentObservation()

        if (currentGameState.isOver()):
            print('GAME IS OVER')
            final_score = CaptureAgent.getScore(self, currentGameState)
            if final_score >0:
                self.won = True
            if (self.terminal and self.won):
                return 10000.  # win is great
            elif final_score ==0:
                return - 500
            else:
                return -1000  # we lost


        # AgentState objects
        # start = startConfiguration
        # configuration = startConfiguration
        # isPacman = boolean
        # scaredTimer = float

        myLastState = lastGameState.getAgentState(self.index)  # Returns AgentState object
        myCurrentState = currentGameState.getAgentState(self.index)  # Returns AgentState object

        # Position
        xLast, yLast = lastGameState.getAgentPosition(self.index)
        xCurr, yCurr = currentGameState.getAgentPosition(self.index)

        # Score information
        lastScore = self.getScore(lastGameState)
        currentScore = self.getScore(currentGameState)
        self.last_score = lastScore  # not used anymore
        self.current_score = currentScore  # not used anymore

        # General food and capsule information
        lastFood = self.getFood(lastGameState)
        lastFoodDefending = self.getFoodYouAreDefending(lastGameState)
        currentFood = self.getFoodYouAreDefending(currentGameState)
        currentFoodDefending = self.getFoodYouAreDefending(currentGameState)
        self.ourFood = self.CountOurFood(currentGameState)  # not used anymore
        self.theirFood = self.CountTheirFood(currentGameState)  # not used anymore

        lastCapsules = self.getCapsules(lastGameState)
        lastCapsulesDefending = self.getCapsulesYouAreDefending(lastGameState)
        currentCapsules = self.getCapsules(currentGameState)
        currentCapsulesDefending = self.getCapsulesYouAreDefending(currentGameState)

        # To check if Pacman ate or dropped or lost food
        lastFoodCarrying = myLastState.numCarrying
        currentFoodCarrying = myCurrentState.numCarrying

        # Check if pacman returned food to our field
        lastFoodReturned = myLastState.numReturned
        currentFoodReturned = myCurrentState.numReturned

        # Add accumulated Reward
        reward = 0

        A = currentFoodCarrying - lastFoodCarrying  # Increase == ate food, Decrease = dropped food || got eaten
        B = currentFoodReturned - lastFoodReturned  # Increase == Dropped food
        C = len(currentCapsulesDefending) - len(lastCapsulesDefending) # Decrease == Enemy ate capsules
        D = len(currentCapsules) - len(lastCapsules)  # Decrease == Ate capsules
        #E = currentFood.count() - lastFood.count()
        F = currentFoodDefending.count() - lastFoodDefending.count() # Decrease == our food eaten, Increase ==  Our Ghost ate pacman || Dropped food
        G = currentScore - lastScore

        if self.isPacman(currentGameState,self.index):
            reward +=1 #get some points for going into enemy territory

        if A > 0:
            reward += A*2  # Eat food
        elif A < 0:
            if B > 0:
                reward += B*10  # Dropped food
            else:
                reward -= 100  # Got eaten ==> Explosion

        if currentGameState.getAgentPosition(self.index) == currentGameState.getInitialAgentPosition(self.index):
                self.atCenter = False
                self.ASTARPATH = self.getCenterPos(currentGameState)
                self.center_counter = 0
                reward -= 100  # we were eaten and spawned back to start

        # if C < 0:
        #     reward -= 5  # Our capsule eaten

        if D < 0:
            reward += 10  # Eat capsule

        # if F < 0:
        #     reward -= F  # Our food eaten
        # elif F > 0:
        #     if B == 0:
        #         reward += 20  # Eat enemy pacman. Not completely correct, becuase other team member might have dropped food

        reward += G

        if reward == 0:
            reward = -1  # Nothing happens, punish time

        return reward

    def observation_step(self, gameState):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(gameState)

            self.last_reward = self.updateLastReward(gameState)  # upate reward
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            if self.params['TRAIN']:
                # Save model
                if (params['save_file']):
                    if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params[
                        'save_interval'] == 0:
                        self.qnet.save_ckpt(
                            'saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))

                        print('Model saved')

                # Train
                self.train()

                self.params['eps'] =  max(self.params['eps_final'], 1.00 - float(self.cnt) / float(self.params['eps_step']))

        # Next
        self.local_cnt += 1
        self.frame += 1

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
        log_file = open('./logs/' + str(self.general_record_time) + '-l-' + str(self.params['width']) + '-m-' + str(
            self.params['height']) + '-x-' + str(self.params['num_training']) + '.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                       (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])  # Why random sampling?
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_n = []  # Next states (s')
            batch_t = []  # Terminal state (t)

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

        if not self.atCenter and self.center_counter == len(self.ASTARPATH) - 1:
            self.atCenter = True
            self.center_counter = 0

        #self.atCenter = True # Always true
        if self.atCenter:
            move = self.getMove(gameState)
        else:
            move = self.ASTARPATH[self.center_counter]
            self.center_counter += 1  # get next move

        # Stop moving when not legal
        legal = gameState.getLegalActions(self.index)

        if move not in legal:
            move = Directions.STOP

        # Save last gameState
        return move

    """Adjusted CODE FROM DQN paper TO GET STATE SPACE for CTF"""

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        size = self.params['STATE_MATRICES']
        total = np.zeros((size, size))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 7
        return total

    def isScared(self, gameState, index):
        """
    Says whether or not the given agent is scared
    """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def isGhost(self, gameState, index):
        """
    Returns true ONLY if we can see the agent and it's definitely a ghost
    """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False

        if not self.red:
            return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))
        else:
            return not ((not gameState.isOnRedTeam(index)) ^ (position[0] >= gameState.getWalls().width / 2))

    def isPacman(self, gameState, index):
        """
    Returns true ONLY if we can see the agent and it's definitely a pacman
    """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False

        if not self.red:
            return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))
        else:
            return not ((not gameState.isOnRedTeam(index)) ^ (position[0] < gameState.getWalls().width / 2))

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

        def getPacmanMatrix(state, who: str):
            """ Return matrix with the player coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)
            cell = 0  # default

            if who == 'Player':
                pos = state.getAgentPosition(self.index)
                if self.isPacman(state, self.index):
                    cell = 1

            elif who == 'Friend':
                team = self.getTeam(state)

                for agent in team:
                    if agent != self.index:
                        pos = state.getAgentPosition(agent)
                        if self.isPacman(state, agent):  # check if friend is pacman or not
                            cell = 1

            elif who == 'Enemy':
                enemies = self.getOpponents(state)

                for agent in enemies:
                    # TODO use probabilities if the

                    # ! TODO check if food is eaten nearby!
                    pos = state.getAgentPosition(agent)
                    if pos is not None:
                        if self.isPacman(state, agent):
                            cell = 1

            else:
                raise TypeError("Need to specify who to check for")

            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state, who: str):
            """ Return matrix with the player coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            if who == 'Player':
                pos = state.getAgentPosition(self.index)
                cell = 0
                if self.isGhost(state, self.index):  # just ghost
                    cell = 1
            elif who == 'Friend':
                team = self.getTeam(state)
                cell =0
                for agent in team:
                    if agent != self.index:
                        pos = state.getAgentPosition(agent)
                        if self.isGhost(state, agent):  # check if friend is pacman or not
                            cell = 1

            elif who == 'Enemy':
                enemies = self.getOpponents(state)
                cell =0
                for agent in enemies:
                    # TODO use probabilities if the
                    pos = state.getAgentPosition(agent)
                    # ! TODO check if food is eaten nearby!
                    if self.isGhost(state, agent):
                        cell = 1
            else:
                raise TypeError("Need to specify who to check for")

            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state, who: str):
            """ Return matrix with the player coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            cell = 0  # default

            if who == 'Player':
                pos = state.getAgentPosition(self.index)

                if self.isScared(state, self.index):  # if we're scared
                    cell = 1

            elif who == 'Friend':
                team = self.getTeam(state)

                for agent in team:
                    if agent != self.index:
                        pos = state.getAgentPosition(agent)
                        if self.isScared(state, agent):  # check if friend is pacman or not
                            cell = 1

            elif who == 'Enemy':
                enemies = self.getOpponents(state)

                for agent in enemies:
                    # ! TODO check if food is eaten nearby!
                    pos = state.getAgentPosition(agent)
                    if pos is not None:
                        if self.isScared(state, agent):
                            cell = 1

            else:
                raise TypeError("Need to specify who to check for")

            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def GetFoodMatrix(state, who: str):
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            if who == 'Defending':  # ours
                grid = self.getFoodYouAreDefending(state)

            elif who == 'Attacking':  # their
                grid = self.getFood(state)

            else:
                raise TypeError("Need to specify what food you want!")

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell

            return matrix

        def GetCapsulesMatrix(state, who: str):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            if who == 'Defending':
                capsules = self.getCapsulesYouAreDefending(state)
            elif who == 'Attacking':
                capsules = self.getCapsules(state)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1 - i[1], i[0]] = 1

            return matrix

        def predictEnemyMatrix(state):
            self.last_food = GetFoodMatrix(state, 'Defending')
            # Check difference from previous
            # if enemy_isempty on our side

            pass

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        # ? 14 matrices
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((self.params['STATE_MATRICES'], height, width))

        # Player info
        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state, 'Player')  # our
        observation[2] = getGhostMatrix(state, 'Player')
        observation[3] = getScaredGhostMatrix(state, 'Player')

        # teammate info
        observation[4] = getPacmanMatrix(state, 'Friend')
        observation[5] = getGhostMatrix(state, 'Friend')
        observation[6] = getScaredGhostMatrix(state, 'Friend')

        # Enemy info
        observation[7] = getPacmanMatrix(state, 'Enemy')
        observation[8] = getGhostMatrix(state, 'Enemy')
        observation[9] = getScaredGhostMatrix(state, 'Enemy')

        # Food and capsules
        observation[10] = GetFoodMatrix(state, 'Defending')
        observation[11] = GetFoodMatrix(state, 'Attacking')
        observation[12] = GetCapsulesMatrix(state, 'Defending')
        observation[13] = GetCapsulesMatrix(state, 'Attacking')

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

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
    Finds the distance between the agent with the given index and its nearest goalPosition
    """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(
            lambda entry: entry[2] + width * height if entry[0] in avoidPositions else 0 + min(
                util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

        # Keeps track of visited positions
        visited = {currentPosition}

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath

class OffensiveAgent(ReflexCaptureAgent):

    def __init__(self, index, *args, **kwargs):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.last_food = CaptureAgent.getFood(self, gameState)
        self.last_capsule = CaptureAgent.getCapsules(self,gameState)
        self.last_enemies = None
        self.last_ghosts = None
        self.get_back_safe = False
        # TODO Add return for food



    def isWall(self,gameState,pos:tuple):
        grid = gameState.data.layout.walls
        return grid[pos[0]][pos[1]]

    def getCenterPos(self, gameState,avoid_enemies=True,run_away=False):
        width = 34
        height = 18

        high =0.75
        low = 0.25

        adjust_x = 1 #quick fix for pre finals
        if run_away:
            adjust_x = 9 # just to get back out of sight
            # current_pos = gameState.getAgentPosition(self.index)
            # if current_pos[1]<int(height/2):
            #     high = 0.8
            #     low = 0.5
            # else:
            #     high = 0.5
            #     low = 0.1

            #if gameState.AgentState.getPosition(self.index)

        # ASTAR Path to center
        pos_to_avoid = []
        if avoid_enemies:
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            ghost = [a for a in enemies if (not a.isPacman) and (a.scaredTimer<=0) and a.getPosition() != None]
            if len(ghost)>0:
                pos_to_avoid = [ghost[0].getPosition()]

        if gameState.isOnRedTeam(self.index):
            pos_x = int(width / 2) - adjust_x
            for i in range(1000):
                pos_y = random.randint(int(height * low), int(high * height))

                center = (pos_x,pos_y)
                if not self.isWall(gameState,center):
                    return deque(self.aStarSearch(gameState.getAgentPosition(self.index), gameState,
                                                  [center],avoidPositions=pos_to_avoid))  # hard code for now
        else: #blue
            pos_x = int(width / 2) + adjust_x
            for i in range(1000):
                pos_y = random.randint(int(height * low), int(high * height))
                center = (pos_x, pos_y)
                if not self.isWall(gameState, center):
                    return deque(self.aStarSearch(gameState.getAgentPosition(self.index), gameState,
                                                  [center],avoidPositions=pos_to_avoid))  # hard code for now


    def path_to_pos(self,gameState,goal_pos:tuple):
        current_pos = gameState.getAgentPosition(self.index)
        return deque(self.aStarSearch(current_pos, gameState, [goal_pos]))


    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
    Finds the distance between the agent with the given index and its nearest goalPosition
    """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(
            lambda entry: entry[2] + width * height if entry[0] in avoidPositions else 0 + min(
                util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

        # Keeps track of visited positions
        visited = {currentPosition}

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()


        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()
        features['foodScore'] = -len(foodList)
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # Computes whether we're on offense (1) or defense (0)
        features['onOffense'] = 0
        if myState.isPacman: features['onOffense'] = 1

        # Compute distance to the nearest pill
        pillList = self.getCapsules(successor)
        if len(pillList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, pill) for pill in pillList])
            features['distanceToPill'] = minDistance

        # Computes distance to threats we can see,
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        threats = [a for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None]
        if len(threats) > 0:
            dists = [len(self.aStarSearch(myPos, gameState,[a.getPosition()])) for a in threats]
            features['threatDistance'] = min(dists)

    # Compute distance to scared ghosts we can see
        ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer != 0 and a.getPosition() != None]
        if len(ghosts) > 0:
            dists = [len(self.aStarSearch(myPos, gameState,[a.getPosition()])) for a in ghosts]
            features['ghostDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 0

        # Food consumption
        carryingFood = myState.numCarrying
        features['carryingFood'] = carryingFood
        features['returnedFood'] = myState.numReturned

        if carryingFood > 5 or len(threats) > 0:
            self.centerPos = self.getCenterPos(gameState)
            self.get_back_safe = True
            #features['closenessToSafety'] = len(centerPos)#self.aStarSearch(myPos, gameState,[centerPos])

        return features

    def getWeights(self, gameState, action):
        return {'foodScore': 50, #previous 100
                'onOffense': 10,
                'distanceToFood': -1,
                'distanceToCapsule': -3,
                'threatDistance': +5,
                'ghostDistance': -10,
                'stop': -100,
                'reverse': -2,
                'carryingFood': -2, # Becomes worse if too many food is carried
                'returnedFood': 10,
                'closenessToSafety': -60
                }

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        if self.get_back_safe:
            if not gameState.getAgentState(self.index).isPacman:
                if len([gameState.getAgentState(i).getPosition()
                    for i in CaptureAgent.getOpponents(self,gameState) if gameState.getAgentState(i).getPosition() is not None])>0: #any opponents nearby?

                    move = self.getCenterPos(gameState,run_away= True)
                    if len(move) > 0:
                        return move.popleft()

                self.get_back_safe = False
            else:
                move = self.getCenterPos(gameState,avoid_enemies=True)
                if len(move)>0:
                    return move.popleft()

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


class terminator(ReflexCaptureAgent):
    """

    This agent should be mainly defensive
    1. if we can find enemy, checks enemies given where candy has been eaten
    2. If none of this, we move towards the most centered capsule to defend

    3. interupt this action as soon as we see an enemy
    4. Maybe stay close to food

    """
    def __init__(self, index, *args, **kwargs):
        CaptureAgent.__init__(self, index)

    # ! 5. What do we do if we're scared
    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.last_food = CaptureAgent.getFoodYouAreDefending(self, gameState)
        self.best_capsule = self.get_best_capsule(gameState)

        self.at_capsule = False
        self.path_to_capsule = self.path_to_pos(gameState,self.best_capsule)
        self.path_to_theif = None #path to the secret theif we cant see (yet)
        self.theif_pos = None
        self.spot_enemies = False # can we see enemies?

    def path_to_pos(self,gameState,goal_pos:tuple):
        current_pos = gameState.getAgentPosition(self.index)
        return deque(self.aStarSearch(current_pos, gameState, [goal_pos]))

    def get_best_capsule(self,gameState):
        """
        A function to get the centermost capsule
        If it's not very centered, we return the center pos/border pos instead
        """
        # First we get the center half the map that has the shortest path to the center
        # Start by just finding the center for us and then see if there's a capsule in there

        # Get capsule map
        # Find Center area in our map
        # See if capsule is in this area
        width, height = gameState.data.layout.width, gameState.data.layout.height

        current_capsules = CaptureAgent.getCapsulesYouAreDefending(self, gameState)
        #map = gameState.data.layout.walls

        center = (int(width/2),int(height/2))
        min = 10**100 #arbirtrary big number
        closest_capsule = None
        for capsule in current_capsules: #find the capsule closest to the middle
            temp = CaptureAgent.getMazeDistance(self, capsule, center)
            if min > CaptureAgent.getMazeDistance(self, capsule, center):
                closest_capsule = capsule
                min = temp
        return closest_capsule


        #? See what centerpos is the most open

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        if not self.at_capsule:
            if len(self.path_to_capsule) == 0:
                self.at_capsule = True
            else:
                action = self.path_to_capsule.popleft()
                return action


        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        if not len(invaders)>0 and self.theif_pos is not None:
            move_to_theif = self.path_to_pos(gameState, self.theif_pos)
            if len(move_to_theif)>0:
                return move_to_theif.popleft()
            else:
                return 'Stop'
        # else:
        #     capsule_action = self.path_to_pos(gameState, self.best_capsule)
        #     if len(capsule_action) > 0:
        #         return capsule_action.popleft()


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

    def outsmart_enemies(self,gameState)->list:
        """Check where the theif is"""
        ourfood = CaptureAgent.getFoodYouAreDefending(self,gameState)
        eaten = None
        if self.last_food.count() != ourfood.count(): #check if there is a food eaten somewhere
            eaten = [(i,k) for i in range(ourfood.width) for k in range(ourfood.height) if
                     self.last_food[i][k] != ourfood[i][k]]

        self.last_food = ourfood # update last food state
        return eaten


    def get_our_center(self, gameState):
        width, height = gameState.data.layout.width, gameState.data.layout.height
        if gameState.isOnRedTeam(self.index):
            x_pos, y_pos = int(width / 4), int(height / 4)
            pass
        else:
            x_pos, y_pos = int(width * (3 / 4)), int(height * (3 / 4))
            pass

    def getFeatures(self, gameState, action):
        """
        Gets the features in the next state, so given the next
        state we see if we got closer to the position we want to be in
        """

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see,
        #! changed to A*
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            self.spot_enemies = True
            dists = [len(self.aStarSearch(myPos, gameState,[a.getPosition()])) for a in invaders]
            features['invaderDistance'] = min(dists)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 0


        #See where enemies eat food, then get the closest food and go there
        missing_food = self.outsmart_enemies(gameState)

        if missing_food is not None:
            all_paths = [[f, self.aStarSearch(myPos, gameState, [f])]
                             for f in missing_food]
            best = min(all_paths)

            self.theif_pos = best[0]
            #if len(best[1]) >0:
            #    self.move_to_theif = deque(best[1]).popleft()


        if self.path_to_theif is not None:
            features['secret_foodtheif'] = len(self.path_to_theif)


        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000,
                'onDefense': 100,
                'invaderDistance': -10,
                'stop': -100,
                'reverse': -2,
                'secret_foodtheif': -15}

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
    Finds the distance between the agent with the given index and its nearest goalPosition
    """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(
            lambda entry: entry[2] + width * height if entry[0] in avoidPositions else 0 + min(
                util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

        # Keeps track of visited positions
        visited = {currentPosition}

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        #print(features, weights)
        #print(action)
        #print(features * weights)
        return features * weights

