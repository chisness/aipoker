@ECHO OFF

IF NOT EXIST build mkdir build

IF NOT EXIST expansion mkdir expansion

IF NOT EXIST expansion\expansion1 mkdir expansion\expansion1

IF NOT EXIST expansion\expansion2 mkdir expansion\expansion2

IF NOT EXIST expansion\expansion3 mkdir expansion\expansion3

IF NOT EXIST expansion\expansion4 mkdir expansion\expansion4

IF NOT EXIST expansion\expansion5 mkdir expansion\expansion5

IF NOT EXIST expansion\expansion6 mkdir expansion\expansion6

IF NOT EXIST bots mkdir bots

IF NOT EXIST data mkdir data

IF NOT EXIST data\cards mkdir data\cards

IF NOT EXIST data\results mkdir data\results

IF NOT EXIST data\serverlog mkdir data\serverlog

javac -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\client\*.java

javac  -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\alien\*.java

javac -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\server\*.java

javac -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\dynamics\*.java

javac  -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\academy25\*.java

javac  -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\tournament\*.java

javac  -cp thirdparty\meerkat-api.jar;thirdparty\smtp.jar;thirdparty\activation.jar;thirdparty\mailapi.jar -d build -sourcepath src src\ca\ualberta\cs\poker\free\alien\*.java

IF NOT EXIST dist mkdir dist


jar cf dist\pokerserver.jar -C build ca


javac -d examples\academy25client -sourcepath examples\academy25client -cp thirdparty\meerkat-api.jar;dist\pokerserver.jar examples\academy25client\TestPlayer.java


jar cf examples\academy25client\TestPlayer.jar -C examples\academy25client TestPlayer.class


copy dist\pokerserver.jar examples\randomclient

copy dist\pokerserver.jar examples\nolimitrandomclient

copy dist\pokerserver.jar examples\academy25client

copy thirdparty\meerkat-api.jar examples\academy25client

copy dist\pokerserver.jar examples\server


jar cf bots\randomclient.jar -C examples randomclient

jar cf bots\nolimitrandomclient.jar -C examples nolimitrandomclient
