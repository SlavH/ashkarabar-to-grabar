@echo off
:: Запуск полного конвейера на disk D:
:: Предварительно скопируйте папку armenian_nlp на D:\

set BASE=D:\armenian_nlp
set SRC=%BASE%\src
set DATA=%BASE%\data
set CKPT=%BASE%\checkpoints
set TOK=%BASE%\tokenizer

echo [1/4] Проверка Python...
python --version || (
    echo Python не найден. Установите с https://python.org
    pause
    exit /b 1
)

echo [2/4] Установка зависимостей...
pip install torch tokenizers sentencepiece tqdm

echo [3/4] Подготовка данных...
python %SRC%\prepare_data.py

echo [4/4] Обучение токенайзера...
python %SRC%\train_tokenizer.py

echo [5/5] Обучение модели...
python %SRC%\train.py

echo.
echo Готово! Чекпоинты в %CKPT%
pause