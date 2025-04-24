@echo off
chcp 65001

:: Remove the requirements.txt file
powershell -Command "Remove-Item -Force requirements.txt"

:: Generate a new requirements.txt file
pip freeze > requirements.txt

:: Remove specific packages from requirements.txt
powershell -Command "(Get-Content requirements.txt) -notmatch 'torch==|torchaudio==|torchvision==' | Set-Content requirements.txt"