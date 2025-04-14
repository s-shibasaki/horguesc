@echo off
REM filepath: d:\work\horguesc\run_pipeline.bat
echo Running Horse Racing Pipeline

echo Step 1: Data Loading...
cd /d %~dp0\Debug
DataLoader.exe

echo Step 2: Training and Prediction...
cd /d %~dp0\x64\Debug
Trainer.exe

echo Pipeline completed.