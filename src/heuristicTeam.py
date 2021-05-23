from myTeam import *
from baselineTeam import ReflexCaptureAgent
import numpy as np
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

        #self.get_best_position(gameState)

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

    def get_best_position(self,gameState):
        """function to find the most open position on our side, that will give us the best cover of the other positions"""
        width, height = gameState.data.layout.width, gameState.data.layout.height
        grid = gameState.getWalls().data
        # Get the relevant part of the grid
        x_dir, y_dir = 10, 18

        if gameState.isOnRedTeam(self.index):
            relevant_grid = grid[9:16]
        else:
            relevant_grid= grid[17:24]

        full_grid = []

        #testx,testy = 0,0
        #best = [len([self.aStarSearch((testx,testy),gameState,[(x,y)]) for x in range(9,17) for y in range(18)]) for testx in range(9,17) for testy in range(18)]
        #best= [[self.aStarSearch((xpos,ypos),gameState,[(x_test,y_test)]) for x_test in ]]
        # Get a centerpoint of that section,
        # compute A* from all other parts to this section
        # Return the one that on averge has the shortest path to the others




        pass
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

