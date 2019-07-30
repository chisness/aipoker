
usage="usage: kuhn <seeds> <numb of hands> <player 1> <player 2> <player 3>"


if [ -z $1 ]
    then
        echo $usage
        exit 1
fi

if [ -z $2 ]
    then
        echo $usage
        exit 1
fi


if [ -z $3 ]
    then 
        echo $usage
        exit 1
fi

if [ -z $4 ]
    then 
        echo $usage
        exit 1
fi

if [ -z $5 ]
    then 
        echo $usage
        exit 1
fi


seed=0
limit=$1
hands=$2
p1=$(echo $3 | cut -d'.' -f 1)
p2=$(echo $4 | cut -d'.' -f 1)
p3=$(echo $5 | cut -d'.' -f 1)
logname="$6""_"$p1"_"$p2"_"$p3"_"$1

if [ -e ../results/$logname ]
    then
        echo "done $logname"
        exit 0
fi

touch ../results/$logname
echo "log name: $logname"
echo "Running tests, seeds: 0 to $limit, numb of hands: $hands, players: $p1 $p2 $p3" 
echo "Running tests, seeds: 0 to $limit, numb of hands: $hands, players: $p1 $p2 $p3"  > ../results/$logname
echo "" >> ../results/$logname

while [ $seed -le $limit ]
do 
    echo "Seed: $seed" >> ../results/$logname
    
    echo "starting position: $p1 $p2 $p3" >> ../results/$logname    
    ./play_match.pl $6"_"$p1"_"$p2"_"$p3"_"$seed ./kuhn.limit.3p.game $2 $seed $p1 ./$3 $p2 ./$4 $p3 ./$5 >> ../results/$logname

    
    echo "starting position: $p1 $p3 $p2" >> ../results/$logname
    ./play_match.pl $6"_"$p1"_"$p3"_"$p2"_"$seed ./kuhn.limit.3p.game $2 $seed $p1 ./$3 $p3 ./$5 $p2 ./$4 >> ../results/$logname

    
    echo "starting position: $p2 $p1 $p3" >> ../results/$logname
    ./play_match.pl $6"_"$p2"_"$p1"_"$p3"_"$seed ./kuhn.limit.3p.game $2 $seed $p2 ./$4 $p1 ./$3 $p3 ./$5 >> ../results/$logname

    
    echo "starting position: $p2 $p3 $p1" >> ../results/$logname
    ./play_match.pl $6"_"$p2"_"$p3"_"$p1"_"$seed ./kuhn.limit.3p.game $2 $seed $p2 ./$4 $p3 ./$5 $p1 ./$3 >> ../results/$logname


    echo "starting position: $p3 $p2 $p1" >> ../results/$logname
    ./play_match.pl $6"_"$p3"_"$p2"_"$p1"_"$seed ./kuhn.limit.3p.game $2 $seed $p3 ./$5 $p2 ./$4 $p1 ./$3 >> ../results/$logname


    echo "starting position: $p3 $p1 $p2" >> ../results/$logname
    ./play_match.pl $6"_"$p3"_"$p1"_"$p2"_"$seed ./kuhn.limit.3p.game $2 $seed $p3 ./$5 $p1 ./$3 $p2 ./$4 >> ../results/$logname

    echo "" >> ../results/$logname
    echo "seed score" >> ../results/$logname    
    python3 ../results/stats.py ../results/$logname $seed >> ../results/$logname  

    echo "accumulated score" >> ../results/$logname    
    python3 ../results/stats.py ../results/$logname >> ../results/$logname  

    echo "" >> ../results/$logname    
    seed=$(($seed+1))
done
