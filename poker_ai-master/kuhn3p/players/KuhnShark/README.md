3-Player Kuhn Agent
===================

author: Adrian Sanchez
e-mail: asanc412@fiu.edu 
team: Amarillo

This is a game-playing agent for 3-player Kuhn Poker that uses nash equilibrium strategies, as described in [http://poker.cs.ualberta.ca/publications/AAMAS13-3pkuhn.pdf](this paper). It is part of a project for the CAP4630 "Intro to Artificial Intelligence" Fall 2017 course.

Most of the template game code is forked from Kevin Waugh's repo, [https://github.com/kdub0/kuhn3p](kuhn3p), but implements a new agent called Shark that plays two Nash equilibrium strategies with probabilites based on which strategy is more effective. This is done so that opponents that implement learning strategies will have a harder time learning from The Shark.

Running the agent is as simple as running `./kuhn_shark.sh <server> <port>` from a terminal window. This script connects to the ACPC dealer using The Shark as an agent. Future modifications to the existing code are being considered to include a learning strategy.
