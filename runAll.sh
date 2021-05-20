#!/bin/bash
#pip install requrements.txt
cd src/
END=6

echo "Playing against baseline team"
for ((i=1;i<=$END;i++));do
    map=$((1 + $RANDOM % 100000));
    echo "Running seed ${map}";
    python capture.py -n 100 -x 400  -r myTeam -b baseline -q -l RANDOM${map}
    python capture.py -n 100 -x 400  -b myTeam -r baseline -q -l RANDOM${map}
done

echo "Playing against heuristic team team"
for ((i=1;i<=$END;i++));do
    map=$((1 + $RANDOM % 100000));
    echo "Running seed ${map}";

    python capture.py -n 100 -x 500  -r myTeam -b heuristicTeam -q -l RANDOM${map}
    python capture.py -n 100 -x 500  -b heuristicTeam -r myTeam  -q -l RANDOM${map}
done
