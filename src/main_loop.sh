#!/usr/bin/env bash
source ~/start.sh
base0=$(ls -1 ../selfplay/*.npz | wc -l)
echo 'Current # of games: '$base0

base1=$((base0 + 250))
base2=$((base1 + 250))
base3=$((base2 + 250))
base4=$((base3 + 250))

python selfplay.py --range $base0 $base1 &
PID1=$!
python selfplay.py --range $base1 $base2 &
PID2=$!
python selfplay.py --range $base2 $base3 &
PID3=$!
python selfplay.py --range $base3 $base4 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4

cd ../selfplay
python scramble.py
cd ../src

tEnd=$base4
vEnd=$((tEnd / 10))
python trainer.py --v_tuple 0 $vEnd --t_tuple $vEnd $tEnd


