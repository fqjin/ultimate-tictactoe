#!/usr/bin/env bash

nohup python selfplay.py --range 6000 6250 &
nohup python selfplay.py --range 6250 6500 &
nohup python selfplay.py --range 6500 6750 &
nohup python selfplay.py --range 6750 7000 &
