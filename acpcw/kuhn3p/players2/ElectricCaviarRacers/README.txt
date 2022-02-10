#############################################################
# Please fill out the following information about your team 
#############################################################

Team Name: 

    Electric Caviar Racers
    
Agent Name (can be the same as team name):

    Electric Caviar

For each team member, please list the following:
Name, Team leader (y/n)?, e-mail, Academic (y/n, position - e.g., PhD student)?, University/Business affiliation, Location (city, province/state, country)

    Eduardo Porto, Yes, eport030@fiu.edu, Yes, undergraduate student, Florida International University, (Miami, FL, US)
    
    Lester Hernandez Alfonso, No, lhern207@fiu.edu, Yes, undergraduate student, Florida International University, (Miami, FL, US)
    

Was this submission part of an academic class project?  What level of class
(undergraduate/graduate)?

    Yes, final project for undergraduate AI class at FIU. Fall 2017.

    
###########################################################################
# Please provide as much information about your agent as possible as the
# competition organizers are very interested in knowing more about the
# techniques used by our competitors.
###########################################################################

1) Is your agent dynamic?  That is, does its strategy change throughout the
course of a match, or is the strategy played the same throughout the match?

    Yes, it bases its predictions by learning opponents strategies. If the opponent to its left is winning more than right, it set a specific probability to them.

    
2) Does your agent use a (approximate) Nash equilibrium strategy?

    Yes, from table 2 & 3 in the article below.

    
3) Does your agent attempt to model your opponents?  If so, does it do so
online during the competition or offline from data (e.g., using logs of play or
the benchmark server)?

    It learns only from each game (of n rounds) it plays.

    
4) Does your agent use techniques that would benefit from additional CPU time
during the competition?

    No. It's learning maxes out at ~5k rounds.

    
5) Does your agent use techniques that would benefit from additional RAM during
the competition?

    No.

    
6) Would you agent benefit from additional disk space?

    No.

    
One/Two Paragraph Summary of Technique

        We implemented the nash equilibrium from the tables 2 & 3 
    in “A Parameterized Family of Equilibrium Profiles for Three-Player 
    Kuhn Poker.”, some of these involed randomizing probabilities.
    The agent was varified against agents with random strategies. 
    Once this was varified we implemented learning. This was done 
    by adjusting the randomized probabilities from the equilibrium to 
    reflect the behaivour of our opponents.

    
References to relevant papers, if any

    Szafron, Duane, et al. “A Parameterized Family of Equilibrium Profiles for Three-Player Kuhn Poker.”     
        poker.cs.ualberta.ca/publications/AAMAS13-3pkuhn.pdf.
