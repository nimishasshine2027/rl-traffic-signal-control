@echo off
cd /d "%~dp0"
start "FIXED" cmd /k python src/demo.py --controller fixed --profile high --gui --delay 60
start "PPO" cmd /k python src/demo.py --controller ppo --profile high --gui --delay 60