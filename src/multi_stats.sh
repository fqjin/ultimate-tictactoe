#!/usr/bin/env bash

python model_stats.py --flag 0 --e 6 &
python model_stats.py --flag 1 --e 6 &
python model_stats.py --flag 2 --e 6 &
python model_stats.py --flag 3 --e 6 &

python model_stats.py --flag 0 --e 5 &
python model_stats.py --flag 1 --e 5 &
python model_stats.py --flag 2 --e 5 &
python model_stats.py --flag 3 --e 5 &

wait

python model_stats.py --flag 0 --e 4 &
python model_stats.py --flag 1 --e 4 &
python model_stats.py --flag 2 --e 4 &
python model_stats.py --flag 3 --e 4 &

python model_stats.py --flag 0 --e 3 &
python model_stats.py --flag 1 --e 3 &
python model_stats.py --flag 2 --e 3 &
python model_stats.py --flag 3 --e 3 &

wait

echo DONE
