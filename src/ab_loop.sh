#!/usr/bin/env bash
source ~/start.sh
echo START
date

base0=20000
base1=$((base0 + 2500))
base2=$((base1 + 2500))
base3=$((base2 + 2500))
base4=$((base3 + 2500))

python selfplayAB.py --range $base0 $base1 &
PID1=$!
python selfplayAB.py --range $base1 $base2 &
PID2=$!
python selfplayAB.py --range $base2 $base3 &
PID3=$!
python selfplayAB.py --range $base3 $base4 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo Selfplay DONE
date

cd ../selfplayAB
zip -u data.zip *.npy
cd ../src

tEnd=$((base4 - 1000))
python trainerAB.py --t_tuple 0 $tEnd --v_tuple $tEnd $base4 --epochs 8
echo Training DONE
date

