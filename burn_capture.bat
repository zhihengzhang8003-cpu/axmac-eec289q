@echo off
"E:\Quart\quartus\bin64\jtagserver.exe" --start
timeout /t 3 /nobreak > nul
"E:\Quart\quartus\bin64\quartus_pgm.exe" --list
"E:\Quart\quartus\bin64\quartus_pgm.exe" -m jtag -o "p;E:\axmac_rtl\rtl\vendor\altera\build\demo\demo.sof"
echo EXIT_CODE=%ERRORLEVEL%
