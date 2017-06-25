@echo off
if not exist ".\out\" mkdir .\out\
cl fracture.cpp^
  -Ox -arch:AVX2 -DEBUG -Z7 -EHsc^
  -Fo:.\out\ -Fe:fracture^
  -I.\third_party\Eigen^
  -link -out:.\out\fracture.exe^