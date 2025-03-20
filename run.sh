#!/bin/bash

# Run main.py with nohup in the background and redirect output to nohup.out
nohup python3 -u main.py > nohup.log 2>&1 &
echo "main.py is running in the background. Check nohup.log for logs."