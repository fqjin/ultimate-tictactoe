#!/usr/bin/env bash

nohup python selfplay.py --range 8000 8250 &
nohup python selfplay.py --range 8250 8500 &
nohup python selfplay.py --range 8500 8750 &
nohup python selfplay.py --range 8750 9000 &
