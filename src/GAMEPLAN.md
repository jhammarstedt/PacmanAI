# TODO for the DQL agent

1. Define states, how will we feed the state to the network: A good could be similar to the state representation as explained in 

## TargetNet and  Q-net
Two nets to be built: 
 * Q- net 
 * TargetNet
 
 Q-net is updated by back prop and update function

 Target net is a early copy of the Q-net which will give the output and is updated every few iterations with the new Qnet

## ML framework 
State representation [this paper](https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf)

<img src = "img\state_space.PNG">


* Reward - Score difference btw our and opposing team
* Network architecture (Conv net with image state space prehaps?)

4. add more