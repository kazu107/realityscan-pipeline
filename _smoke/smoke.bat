@echo off
REM --- RealityScan headless smoke test -------------------------------------
REM 80 images (frames 0000-0009 x 8 views) -> align -> max component -> export
set RS="C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
set W=K:\realityscan\_smoke
set OUT=%W%\out

if not exist "%OUT%" mkdir "%OUT%"
if not exist "%W%\crash" mkdir "%W%\crash"

%RS% -headless ^
     -stdConsole ^
     -silent "%W%\crash" ^
     -set "appQuitOnError=true" ^
     -set "appAutoSaveMode=false" ^
     -set "appIncSubdirs=false" ^
     -writeProgress "%W%\progress.txt" 10 ^
     -newScene ^
     -add "%W%\test.imagelist" ^
     -selectAllImages ^
     -editInputSelection "inpCalibration=1" ^
     -editInputSelection "inpFocal=15.1039" ^
     -editInputSelection "inpDistortionModel=0" ^
     -editInputSelection "inpDistortion=2" ^
     -deselectAllImages ^
     -tag "ALIGN_BEGIN" ^
     -align ^
     -tag "ALIGN_END" ^
     -selectMaximalComponent ^
     -tag "COMPONENT_SELECTED" ^
     -exportSparsePointCloud "%OUT%\sparse.ply" ^
     -tag "PLY_DONE" ^
     -exportRegistration "%OUT%\transforms.json" ^
     -tag "REG_DONE" ^
     -exportSelectedComponentFile "%OUT%\component.rsalign" ^
     -tag "COMP_DONE" ^
     -save "%W%\smoke.rsproj" ^
     -quit

echo EXITCODE=%errorlevel%
