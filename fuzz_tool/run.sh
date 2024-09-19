#!/bin/bash

cd fuzz_tool
python3 ./src/main.py --opt=generator  --sqlName=test
wait
python3 ./src/main.py --opt=fuzz  --sqlName=test








