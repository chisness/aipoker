IF "%2"=="" GOTO ERROR

java -cp meerkat-api.jar;pokerserver.jar;TestPlayer.jar TestPlayer %1 %2


GOTO END

:ERROR
@ECHO Usage: startme.bat <ipaddress> <portnumber>

:END