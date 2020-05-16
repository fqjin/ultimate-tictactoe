#!/usr/bin/env bash

nohup python selfplay.py --range 6000 6125 &
nohup python selfplay.py --range 6125 6250 &
nohup python selfplay.py --range 6250 6375 &
nohup python selfplay.py --range 6375 6500 &
nohup python selfplay.py --range 6500 6625 &
nohup python selfplay.py --range 6625 6750 &
nohup python selfplay.py --range 6750 6875 &
nohup python selfplay.py --range 6875 7000 &
