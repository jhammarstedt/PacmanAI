
<img src="http://ai.berkeley.edu/images/pacman_game.gif" width=350 height = 200 align ="right" />

# Capture that flag Pacman

## Authors ##
* Johan Hammarstedt, [jhammarstedt](https://github.com/jhammarstedt)
* Lukas Vordemann, [vordemann](https://github.com/vordemann)

## Task
To build a team to play CTF- Pacman, created by the competition hosted at [Berkley](http://ai.berkeley.edu/contest.html). This was built as a school assignment for the class [DD2438](https://www.kth.se/student/kurser/kurs/DD2438?l=en) at KTH with an inter-class tournament, which we ended up winning.
![](http://ai.berkeley.edu/projects/release/contest/v1/002/capture_the_flag.png)
### Initial idea
Our orignal idea was to train a DQL agent to play capture the flag pacman using CNN + fully connected layers. However, the setup for the agent is there, but due to the short deadline we didn't have time to train it sufficiently and as a result very little learning was done.

### Final team
As a backup to the DQL agent we also added a heuristic based team, that ended up performing really well and won the final tournament.
Setup:
* Defensive Agent:
  * Uses A* to get all paths
  * Tracks enemy movement by checking where food disapears (since you don't have vision of enemies if they're too far away)
  * Rush to capsules at beginning to prevent it from being stolen
* Offensive Agent:
  * Find closest pill with A*
  * "Risk Assesment", will only eat aroud 5 foods at the time to avoid overextending
  * If chased it will compute the best path to base using A*, or check for a closer capsule while avoiding enemies
  * A flanking manouver to avoid getting stuck with enemies

## Run
Use the following command to run the code for the heuristic vs a baseline team
```$ python capture.py -r heuristicTeam -b baseline -l RANDOM50 ```

To train the DQN agent:
```$ python capture.py -r DQN_agent -b baseline -n 5000 -x 4000 -l RANDOM99```
* `-n` is number of episodes
* `-x` is number of trainings
* `-l` choose a random grid


## Game enviroment 
Credits to cshelton for rewriting the enviroment to support python 3, you can find the repository [here](https://github.com/cshelton/pacman-ctf)

## Useful resources:
 * [CFT Pacman](https://github.com/jaredjxyz/Pacman-Tournament-Agent)
 * [starcraft RL](https://soygema.github.io/starcraftII_machine_learning/#0)
 * RL in pytorch: [Mario](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html), [Flappy Bird](https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial)
 * [Keras RL](https://github.com/keras-rl/keras-rl)
 * [MARL on github](https://github.com/topics/multiagent-reinforcement-learning)
 * [Pacman DQL](https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf), [repo](https://github.com/tychovdo/PacmanDQN)

