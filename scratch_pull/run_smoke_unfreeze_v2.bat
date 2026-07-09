@echo off
cd /d C:\Users\islab01\vla-atlas\experiments\firstpass_sweep
set CUDA_VISIBLE_DEVICES=1
C:\Users\islab01\vla-atlas\venv\Scripts\python.exe smoke_unfreeze_vision_v2.py >> smoke_unfreeze_vision_v2_stdout.log 2>&1
