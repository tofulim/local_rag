#!/bin/bash

git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-cpu.txt
pip install -e .
