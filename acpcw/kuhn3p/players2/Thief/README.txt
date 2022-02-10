Agent Name: Theif

Team Members: Robert Panoff, Carolina Uribe-Gosselin

3-Player Kuhn Poker Agent

How to Run:

Execute the startme.sh script located in the root directory.This script takes two parameters, the server address and the port.

startme.sh addr port

The script executes the python file connect_to_dealer.py, which connects to dealer.c from the project_acpc_server directory (project_acpc_server/dealer.c).

Additional Notes:
The Theif agent, when created, takes in a single arguement (a number between 0 and 1) to determine it's baseline bluff value. This value will be modified as it plays games, but a high number is likely to stay high.