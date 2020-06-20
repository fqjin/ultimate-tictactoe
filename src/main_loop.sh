#!/usr/bin/env bash
source ~/start.sh
base0=$(ls -1 ../selfplay/*.npz | wc -l)
echo 'Current # of games:' $base0
echo START
date

base1=$((base0 + 1000))
base2=$((base1 + 1000))
base3=$((base2 + 1000))
base4=$((base3 + 1000))

python selfplay.py --range $base0 $base1 &
PID1=$!
python selfplay.py --range $base1 $base2 &
PID2=$!
python selfplay.py --range $base2 $base3 &
PID3=$!
python selfplay.py --range $base3 $base4 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo Selfplay DONE
date

cd ../selfplay
python scramble.py
cd ../src

tEnd=$base4
vEnd=$((tEnd / 10))
python trainer.py --v_tuple 0 $vEnd --t_tuple $vEnd $tEnd --epochs 12
rm -r ../selfplay/scramble
echo Training DONE
date

python model_stats.py --flag 0 --e 10 &
python model_stats.py --flag 1 --e 10 &
python model_stats.py --flag 2 --e 10 &
python model_stats.py --flag 3 --e 10 &
wait

python model_stats.py --flag 0 --e 9 &
python model_stats.py --flag 1 --e 9 &
python model_stats.py --flag 2 --e 9 &
python model_stats.py --flag 3 --e 9 &
wait

python model_stats.py --flag 0 --e 8 &
python model_stats.py --flag 1 --e 8 &
python model_stats.py --flag 2 --e 8 &
python model_stats.py --flag 3 --e 8 &
wait

python model_stats.py --flag 0 --e 7 &
python model_stats.py --flag 1 --e 7 &
python model_stats.py --flag 2 --e 7 &
python model_stats.py --flag 3 --e 7 &
wait

python model_stats.py --flag 0 --e 6 &
python model_stats.py --flag 1 --e 6 &
python model_stats.py --flag 2 --e 6 &
python model_stats.py --flag 3 --e 6 &
wait

python model_stats.py --flag 0 --e 5 &
python model_stats.py --flag 1 --e 5 &
python model_stats.py --flag 2 --e 5 &
python model_stats.py --flag 3 --e 5 &
wait

echo Testing DONE
date
