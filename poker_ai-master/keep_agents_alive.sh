while true
do
    python2 newuak_dealer.py 127.0.0.1 30001 v2 &
    python2 newuak_dealer.py 127.0.0.1 30002 v3

    sleep 0.5
done
