#!/usr/bin/env bash

nohup python selfplay.py --range 10000 10250 &
nohup python selfplay.py --range 10250 10500 &
nohup python selfplay.py --range 10500 10750 &
nohup python selfplay.py --range 10750 11000 &
