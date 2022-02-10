kuhn3p
======

author: Kevin Waugh
e-mail: kevin.waugh@gmail.com

The submission deadline for the 2014 3-player Kuhn poker competition has been
extended to July 5th.

To get you started, I've written this python package to accelerate the
development of 3-player Kuhn programs.  Feel free to copy and use any of the
code here to make your own player.  

Recall that for this competition, we are asking you to submit code for your
player.  If you build a program that relies on some ahead of time computation,
this does not have to be submitted (though we encourage you to).

The rules for the competition are on the
[http://www.computerpokercompetition.org/](Annual Computer Poker Competition)
website.  In short, you are allowed 10MB of uncompressed disk space and may
use up to 1 second per 100 hands.  That works out to 30 seconds per match.

Getting started
---------------

You can run a 3000 hand match by: `python run\_match.py`

Initially, it will run a match between always raise, always call, and a bot
that bluffs 20% of the time.

To change the bots you would like to run a match with, edit the\_players in
run\_match.py

For an idea about how to create your own player, see kuhn3p/Players/Bluffer.py

What's coming
-------------

I plan on writing a few more sophistocated players in the near future.  For
example, one trained using counterfactual regret minimization, and another that
solves for a correlated equilibrium.

At this time, this package does not communicate with the ACPC dealer package.
I will be adding the hooks to do this.  That is, if you write your bot to
conform to the kuhn3p.Player interface, then it will be able to run against
bots written to interact with the regular dealer.

I will also be adding code to run a tournament within this package.  You
will be able to choose the bots you enter, and it will run the matches
required to get statistical significance.
