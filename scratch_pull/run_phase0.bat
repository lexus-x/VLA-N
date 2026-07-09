@echo off
cd /d C:\Users\islab01\vla-atlas\experiments\firstpass_sweep
set CUDA_VISIBLE_DEVICES=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C:\Users\islab01\vla-atlas\venv\Scripts\python.exe phase0_validate.py >> phase0_stdout.log 2>&1
