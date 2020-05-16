#!/usr/bin/env bash

nohup python selfplay.py --range 9000 9250 &
nohup python selfplay.py --range 9250 9500 &
nohup python selfplay.py --range 9500 9750 &
nohup python selfplay.py --range 9750 10000 &
