*** CONTENTS ***
This package contains:
1. The server code for the AAAI Computer Poker Competition.
2. Client examples for the AAAI CPC.
3. Code for the benchmark suite.

*** INSTALLATION ***
For Linux, see ReadMeLinux.txt.

In order to install this software:
1. Install cygwin and a java SDK. Cygwin is necessary only if you
want to use tarfiles. Make sure to include ssh (in the Net->openssh package)
for the server and sshd for the client.

2. Make sure the PATH includes the bin directory of both
java and cygwin.

3. Run compile.bat

4. To test the server, run run.bat (for limit) or runNoLimit.bat (for no-limit).

*** EXAMPLES ***
A few examples, a basic java client, a poker academy client, and a server for them
to connect to, are located in the examples directory. The first two are examples
of what should be submitted to the competition. The README.txt in that directory
goes into more detail.

This zip file contains the source code in Java and documentation for a freeware Poker 
Server and some sample client bots. It is recommended that the user extend one of
these clients for his or her purposes unless he or she has an existing Poker Academy
bot.

compile.bat: recompiles the client and the server.
run.bat: runs the server with the basic (random) client together on the localhost.
runNoLimit.bat: runs the server with a basic (no-limit random) client together on the localhost.

It also contains a wrapper client that can connect to a Meerkat API bot.

If you want to run your own (limit) bot, then you can modify winlocal.prf and
change Random1 or Random2 to point to your own jar or tar file. See the comments in winlocal.prf
for more details.

To run a no-limit bot, you can modify nolimitwinlocal.prf and execute runNoLimit.bat.



