@echo off
echo Starting JTAG server...
"E:\Quart\quartus\bin64\jtagserver.exe" --start

echo Waiting for JTAG server...
timeout /t 3 /nobreak > nul

echo Detecting hardware...
"E:\Quart\quartus\bin64\quartus_pgm.exe" --list

echo Programming FPGA...
"E:\Quart\quartus\bin64\quartus_pgm.exe" -m jtag -o "p;E:\axmac_rtl\rtl\vendor\altera\build\demo\demo.sof"

echo.
echo Done. Check above for success/error.
pause
