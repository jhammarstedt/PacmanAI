class DQN_agent(game.agent):
    """The agent we will use later"""

    def __init__(self):
        x = pass 



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
                        matrix[-1-i][j] = cell
                return matrix

            def getPacmanMatrix(state):
                """ Return matrix with pacman coordinates set to 1 """
                width, height = state.data.layout.width, state.data.layout.height
                matrix = np.zeros((height, width), dtype=np.int8)

                for agentState in state.data.agentStates:
                    if agentState.isPacman:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

                return matrix

            def getGhostMatrix(state):
                """ Return matrix with ghost coordinates set to 1 """
                width, height = state.data.layout.width, state.data.layout.height
                matrix = np.zeros((height, width), dtype=np.int8)

                for agentState in state.data.agentStates:
                    if not agentState.isPacman:
                        if not agentState.scaredTimer > 0:
                            pos = agentState.configuration.getPosition()
                            cell = 1
                            matrix[-1-int(pos[1])][int(pos[0])] = cell

                return matrix

            def getScaredGhostMatrix(state):
                """ Return matrix with ghost coordinates set to 1 """
                width, height = state.data.layout.width, state.data.layout.height
                matrix = np.zeros((height, width), dtype=np.int8)

                for agentState in state.data.agentStates:
                    if not agentState.isPacman:
                        if agentState.scaredTimer > 0:
                            pos = agentState.configuration.getPosition()
                            cell = 1
                            matrix[-1-int(pos[1])][int(pos[0])] = cell

                return matrix

            def getFoodMatrix(state):
                """ Return matrix with food coordinates set to 1 """
                width, height = state.data.layout.width, state.data.layout.height
                grid = state.data.food
                matrix = np.zeros((height, width), dtype=np.int8)

                for i in range(grid.height):
                    for j in range(grid.width):
                        # Put cell vertically reversed in matrix
                        cell = 1 if grid[j][i] else 0
                        matrix[-1-i][j] = cell

                return matrix

            def getCapsulesMatrix(state):
                """ Return matrix with capsule coordinates set to 1 """
                width, height = state.data.layout.width, state.data.layout.height
                capsules = state.data.layout.capsules
                matrix = np.zeros((height, width), dtype=np.int8)

                for i in capsules:
                    # Insert capsule cells vertically reversed into matrix
                    matrix[-1-i[1], i[0]] = 1

                return matrix

            # Create observation matrix as a combination of
            # wall, pacman, ghost, food and capsule matrices
            # width, height = state.data.layout.width, state.data.layout.height 
            width, height = self.params['width'], self.params['height']
            observation = np.zeros((6, height, width))

            observation[0] = getWallMatrix(state)
            observation[1] = getPacmanMatrix(state)
            observation[2] = getGhostMatrix(state)
            observation[3] = getScaredGhostMatrix(state)
            observation[4] = getFoodMatrix(state)
            observation[5] = getCapsulesMatrix(state)

            observation = np.swapaxes(observation, 0, 2)

            return observation