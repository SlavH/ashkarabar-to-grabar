@echo off
REM Test the Armenian translation model on disk D:
REM Usage: run_tests.bat [single|batch|eval]

set BASE=D:\armenian_nlp
set SRC=%BASE%\src

if not exist "%BASE%\checkpoints\armenian_mt.pt" (
    echo ERROR: Model checkpoint not found at %BASE%\checkpoints\armenian_mt.pt
    echo Make sure you copied the Kaggle output to D:\armenian_nlp\
    pause
    exit /b 1
)

if "%1"=="" (
    echo Usage: run_tests.bat [single|batch|eval]
    echo.
    echo   single  - Test one sentence (prompts for input)
    echo   batch   - Test first 20 validation pairs
    echo   eval    - Compute chrF and BLEU metrics (requires sacrebleu)
    pause
    exit /b 0
)

if "%1"=="single" (
    python %SRC%\..\test_one.py --text "%2"
) else if "%1"=="batch" (
    python %SRC%\..\test_batch.py %2
) else if "%1"=="eval" (
    python %SRC%\..\eval_metrics.py
) else (
    echo Unknown option: %1
)
