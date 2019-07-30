#!/bin/bash
if [ "$#" -ne 2 ]; then
	echo "Usage: ./kuhn_shark.sh <server> <port>"
	exit
fi

python connect_to_dealer.py $1 $2
