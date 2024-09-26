@echo off
set /p CommitMessage=Enter commit message: 
git add .
git commit -m "%CommitMessage%"
git push origin main
