#!/bin/bash
cd /home/dietpi/Snake2.0-AI
nohup .venv/bin/python -u run_curriculum.py > run_curriculum.out 2>&1 &
echo "PID=$!" >> run_curriculum.out 2>&1
