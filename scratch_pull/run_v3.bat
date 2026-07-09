@echo off
cd /d C:\Users\islab01\vla-atlas\experiments\firstpass_sweep
set CUDA_VISIBLE_DEVICES=1
C:\Users\islab01\vla-atlas\venv\Scripts\python.exe decoder_fix_validation_v3.py >> decoder_fix_validation_v3_stdout.log 2>&1
