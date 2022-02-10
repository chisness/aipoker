IF "%2"=="" GOTO ERROR
java -cp meerkat-api.jar;pokerserver.jar ca.ualberta.cs.poker.free.client.NoLimitRandomPokerClient %1 %2

GOTO END

:ERROR
@ECHO Usage: startme.bat <ipaddress> <portnumber>

:END