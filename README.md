
<img src="http://ai.berkeley.edu/images/pacman_game.gif" width=350 height = 200 align ="right" />

# PacmanAI (WIP)

## Authors ##
* Johan Hammarstedt, [jhammarstedt](https://github.com/jhammarstedt)
* Lukas Vordemann, [vordemann](https://github.com/vordemann)

## Task

We will train an RL agent to play capture-the-flag pacman inspired by the competition at [Berkley](http://ai.berkeley.edu/contest.html)

This is a task for the class [DD2438](https://www.kth.se/student/kurser/kurs/DD2438?l=en), expected to be done before June.

![](http://ai.berkeley.edu/projects/release/contest/v1/002/capture_the_flag.png)

## Game enviroment 
Credits to cshelton for rewriting the enviroment to support python 3, you can find the repository [here](https://github.com/cshelton/pacman-ctf)

## Useful resources:
 * [CFT Pacman](https://github.com/jaredjxyz/Pacman-Tournament-Agent)
 * [starcraft RL](https://soygema.github.io/starcraftII_machine_learning/#0)
 * RL in pytorch: [Mario](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html), [Flappy Bird](https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial)
 * [Keras RL](https://github.com/keras-rl/keras-rl)
 * [MARL on github](https://github.com/topics/multiagent-reinforcement-learning)
 * [Pacman DQL](https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf)

## Run
Use the following command to run the code
'''$ python capture.py -n 6000 -x 5000 '''
* -n is number of episodes
* -x is number of trainings
* -l smallGrid (Optional)