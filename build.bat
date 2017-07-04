@echo off

if not exist ".\out\" mkdir .\out\

cl fracture.cpp^
  mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib^
  -Ox -arch:AVX2 -fp:fast -GL^
  -DEBUG -Z7^
  -EHsc -MT^
  -Fo:.\out\ -Fe:fracture^
  -I.\third_party\Eigen^
  -link -out:.\out\fracture.exe -LTCG^