@echo off
setlocal

set MASTER=../../results/master_results.csv
set DOCS=../../docs

for %%f in (make_*.py) do (
    echo ============================================
    echo Running %%f
    python %%f --master_csv %MASTER% --docs_dir %DOCS
    if errorlevel 1 (
        echo ERROR while running %%f
        exit /b 1
    )
)

echo ============================================
echo All make_*.py scripts executed successfully.
endlocal
