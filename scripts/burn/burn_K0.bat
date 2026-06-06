@echo off
echo Starting JTAG server...
"E:\Quart\quartus\bin64\jtagserver.exe" --start
timeout /t 3 /nobreak > nul
echo Programming K0_trunc...
"E:\Quart\quartus\bin64\quartus_pgm.exe" -m jtag -o "p;E:\uart_builds\K0_trunc\demo.sof"
echo.
echo Done. LED should show argmax in binary.
pause
