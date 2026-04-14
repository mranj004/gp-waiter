@echo off
setlocal
if "%~6"=="" (
  echo Usage:
  echo   run_train_windows.bat ^<genotype_txt^> ^<phenotype_csv^> ^<weight_csv^> ^<phenotype_column^> ^<rows^> ^<cols^>
  exit /b 1
)

cd /d "%~dp0"
python train-test.py --genotype-txt "%~1" --phenotype-csv "%~2" --weight-csv "%~3" --phenotype-column "%~4" --rows %5 --cols %6
endlocal
