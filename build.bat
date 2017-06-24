@echo off
if not exist ".\out\" mkdir .\out\
cl /Ox /arch:AVX2 /DEBUG /Z7 /Fo.\out\ /Fefracture /EHsc fracture.cpp /link /out:.\out\fracture.exe