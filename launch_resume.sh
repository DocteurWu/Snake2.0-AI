#!/bin/bash
cd /home/dietpi/Snake2.0-AI
export RESUME=1
nohup .venv/bin/python -u run_24h.py > run_24h.out 2>&1 &
echo "PID=$!" >> run_24h.out 2>&1
