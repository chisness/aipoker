Poker Server software linux notes
---------------------------------

Much of this guide is now at: http://www.cs.ualberta.ca/~pokert/2007/docs.php

Note: The following is a guide for using the server to run your own 
tournaments.If you want to connect to the benchmark server and run 
matches against the 2006 or 2007 competitors, see below.

To compile the server, and create the necessary directories, run
compile.sh. 

To run a tournament you will also need to execute: run.sh after setting
up the profile, linuxlocal.prf

If running the server on linux be sure to have absolute paths in the
profile. See linuxlocal.prf for an example. Paths relative to the
$HOME directory cause problems, and absolute paths reduce the
complexity of keeping everything straight.

The server supports tar files as well as jar files, but make sure you
have a startme.sh file in your tar/jar archive, with the proper
permissions for execute (chmod a+x startme.sh if needed)

A mix of linux and windows machines is possible, as long as the
profile is correct. Since most linux machines will have sshd running,
just be sure to have ssh-keys setup for passwordless logins if you
plan on using remote machines. 



QuickStart
----------

From the root directory:

1. compile.sh
2. COPY your bots to ./bots/YOURBOT in the tarfile or jarfile, make sure you have a startme.sh script to start your bot in the archive
3. Edit linuxlocal.prf or linuxremote.prf with the details of the bots
in the tournament, and the machines being used
4. If using remote machine for matches, setup passwordless ssh keys
for each remote machine, and login once once to avoid the prompt
5. ./run.sh 


Benchmark Server Access
-----------------------
1. ./compile.sh
2. COPY your bots to ./bots/YOURBOT in the tarfile or jarfile, make sure you have a startme.sh script to start your bot in the archive
3. Edit graphicalalienclientlimit2006.prf,graphicalalienclientlimit2007.prf or graphicalalienclientlimit2007.prf if you bot is limit, nolimit or was created for the older 2006 limit tournament. Add the definition of your bot, and the path to the directory to expand your bot into. Fill in the YOURPATH fields.
5. ./runGraphical[NoLimit2007/Limit2007/Limit2006].sh
6. Login with your username and password, if you don't have one email myself or Martin, contact information is available on the website.


--
Christian Smith
chsmith@cs.ualberta.ca
