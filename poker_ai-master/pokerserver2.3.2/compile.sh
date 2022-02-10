#!/bin/bash
#
# Linux compile script for the poker server
# After running compile.sh, look for a run.sh script
# to avoid typing out a classpath
# Author: Christian Smith


if [ ! -e "build" ]; 
    then mkdir build 
fi

if [ ! -e "expansion" ]; 
    then mkdir expansion 
fi

if [ ! -e "expansion/expansion1" ];
    then mkdir expansion/expansion1
fi

if [ ! -e "expansion/expansion2" ];
    then mkdir expansion/expansion2
fi

if [ ! -e "expansion/expansion3" ];
    then mkdir expansion/expansion3
fi

if [ ! -e "expansion/expansion4" ];
    then mkdir expansion/expansion4
fi

if [ ! -e "expansion/expansion5" ];
    then mkdir expansion/expansion5
fi

if [ ! -e "expansion/expansion6" ];
    then mkdir expansion/expansion6
fi

if [ ! -e "bots" ]; 
    then mkdir  bots 
fi
if [ ! -e "data" ]; 
    then mkdir data 
fi

if [ ! -e "data/cards" ]; 
    then mkdir data/cards
fi

if [ ! -e "data/results" ]; 
    then mkdir data/results
fi

if [ ! -e "data/serverlog" ]; 
    then mkdir data/serverlog
fi


javac -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/client/*.java

javac  -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/alien/*.java

javac -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/server/*.java

javac -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/dynamics/*.java

javac  -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/academy25/*.java

javac  -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/tournament/*.java

javac  -cp thirdparty/meerkat-api.jar:thirdparty/smtp.jar:thirdparty/activation.jar:thirdparty/mailapi.jar -d build -sourcepath src src/ca/ualberta/cs/poker/free/alien/*.java

echo Done Compiling

if [ ! -e "dist" ]; 
    then mkdir dist
fi

jar cf dist/pokerserver.jar -C build ca

javac -d examples/academy25client -sourcepath examples/academy25client -cp thirdparty/meerkat-api.jar:dist/pokerserver.jar examples/academy25client/TestPlayer.java

jar cf bots/randomclient.jar -C examples randomclient

jar cf examples/academy25client/TestPlayer.jar -C examples/academy25client TestPlayer.class


cp dist/pokerserver.jar examples/randomclient

cp dist/pokerserver.jar examples/nolimitrandomclient

cp dist/pokerserver.jar examples/academy25client

cp thirdparty/meerkat-api.jar examples/academy25client

cp dist/pokerserver.jar examples/server


tar cf bots/randomclient.tar -C examples randomclient

jar cf bots/nolimitrandomclient.jar -C examples nolimitrandomclient

