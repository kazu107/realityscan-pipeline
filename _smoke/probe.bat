@echo off
REM --- probe: report variables + params.xml driven registration export ------
set RS="C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
set W=K:\realityscan\_smoke
set OUT=%W%\probe_out
set P=%W%\params

if not exist "%OUT%" mkdir "%OUT%"

%RS% -headless ^
     -stdConsole ^
     -silent "%W%\crash" ^
     -set "appQuitOnError=true" ^
     -set "appAutoSaveMode=false" ^
     -load "%W%\smoke.rsproj" ^
     -selectMaximalComponent ^
     -printReport "RSSTAT name=$(componentName) cameras=$(componentCamerasCount) points=$(componentPointsCount) meanErr=$(componentMeanError:.4f) medianErr=$(componentMedianError:.4f) maxErr=$(componentMaximalError:.4f) track=$(componentAverageTrackLength:.2f) imagesInProject=$(imageCount) guid=$(componentGUID)" ^
     -tag "REPORT_DONE" ^
     -exportRegistration "%OUT%\cameras_opencv.csv" "%P%\reg_opencv_csv.xml" ^
     -tag "OPENCV_CSV_DONE" ^
     -exportRegistration "%OUT%\cameras_intext.csv" "%P%\reg_intext_csv.xml" ^
     -tag "INTEXT_CSV_DONE" ^
     -exportSparsePointCloud "%OUT%\sparse.ply" "%P%\sparse_ply.xml" ^
     -tag "PLY_PARAMS_DONE" ^
     -quit

echo EXITCODE=%errorlevel%
