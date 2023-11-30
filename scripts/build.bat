:: Windows build script
@echo off

cd ..
if not exist "build" mkdir build
if not exist "bin" mkdir bin

cd build

cmake -G "Visual Studio 15 2017 Win64" -T v140 ..
cmake --build . -- /M

cd ..\scripts
