@echo off
del graph*.dot* 2>nul
del *.csv 2>nul
rd /q /s .\data\ 2>nul
rd /q /s .\graphs\ 2>nul
rd /q /s .\pdf\ 2>nul
rd /q /s .\results\ 2>nul