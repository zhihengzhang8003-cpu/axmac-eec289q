@echo off
setlocal

set VLIB=E:\Quart\modelsim_ase\win32aloem\vlib.exe
set VMAP=E:\Quart\modelsim_ase\win32aloem\vmap.exe
set VLOG=E:\Quart\modelsim_ase\win32aloem\vlog.exe
set VSIM=E:\Quart\modelsim_ase\win32aloem\vsim.exe

set TBDIR=E:\uart_builds\drum_tb
set GOLDENDIR=E:\uart_builds\golden

echo === Setting up build area at %TBDIR% ===
if not exist "%TBDIR%"     mkdir "%TBDIR%"
if not exist "%GOLDENDIR%" mkdir "%GOLDENDIR%"

copy /Y "E:\AIDev\EEC 289Q 002 SQ 2026：Deep Learning Hardware\project\rtl\src\drum_multiplier.v"    "%TBDIR%\drum_multiplier.v"    >nul 2>&1
copy /Y "E:\AIDev\EEC 289Q 002 SQ 2026：Deep Learning Hardware\project\rtl\tb\tb_drum_multiplier.sv" "%TBDIR%\tb_drum_multiplier.sv" >nul 2>&1
copy /Y "E:\AIDev\EEC 289Q 002 SQ 2026：Deep Learning Hardware\project\rtl\golden\drum_int8.csv"     "%GOLDENDIR%\drum_int8.csv"     >nul 2>&1

echo Checking copies:
dir /b "%TBDIR%\drum_multiplier.v"    2>&1
dir /b "%TBDIR%\tb_drum_multiplier.sv" 2>&1
dir /b "%GOLDENDIR%\drum_int8.csv"    2>&1

cd /d "%TBDIR%"
echo Current dir: %CD%

echo.
echo === Creating work library ===
"%VLIB%" work
if errorlevel 1 echo vlib failed, continuing...

echo.
echo === Mapping library ===
"%VMAP%" work work

echo.
echo === Compiling ===
"%VLOG%" -sv "%TBDIR%\drum_multiplier.v" "%TBDIR%\tb_drum_multiplier.sv"
if errorlevel 1 (
    echo *** COMPILE FAILED ***
    exit /b 1
)

echo.
echo === Simulating ===
"%VSIM%" -c tb_drum_multiplier -do "run -all; quit -f"

endlocal
