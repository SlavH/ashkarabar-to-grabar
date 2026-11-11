# Armenian NLP: Ашхарабар → Грабар (WORK ONLY ON DISK D:)

Все ресурсы берутся с Kaggle. Модель обучается на GPU Kaggle, чекпоинты сохраняются на D:.

## Структура на D:\

```
D:\armenian_nlp\
├── data\
│   ├── raw_pairs.txt    # ашхарабар\tграбар (создайте вручную или скачайте с Kaggle)
│   ├── train.csv
│   └── val.csv
├── checkpoints\         # веса модели (лучшая и последняя версия)
├── tokenizer\           # sp_model.json (BPE токенайзер)
├── src\
│   ├── prepare_data.py  # очистка и разбивка данных
│   ├── train_tokenizer.py # обучение BPE токенайзера
│   ├── model.py         # Encoder-Decoder Transformer с нуля
│   ├── train.py         # цикл обучения с чекпоинтами
│   └── infer.py         # инференс / перевод
├── kaggle_train.ipynb   # ноутбук для обучения на Kaggle GPU
└── run_on_D.bat         # полный конвейер на Windows
```

## Быстрый старт

### 1. Скопируйте папку на D:
Скопируйте всю папку `armenian_nlp` на `D:\armenian_nlp\`.

### 2. Подготовьте данные
Файл `data/raw_pairs.txt` должен содержать пары:
```
ашхарабарное предложение	грабарное предложение
по одной паре на строку, разделение табуляцией
```

Минимум для старта: 1000 пар. Желательно 5000+.

Куда брать данные:
- Kaggle: `ffatty/plaintext-wikipedia-full-armenian` (современный армянский)
- Kaggle: `alvations/old-newspapers` (старые газеты,可能 содержат grabar)
- Самостоятельно составить пары (2000–5000 предложений)

### 3. Установите зависимости
```cmd
D:
cd D:\armenian_nlp
pip install torch tokenizers sentencepiece tqdm
```

### 4. Запустите обучение
```cmd
run_on_D.bat
```

Или по шагам:
```cmd
python src\prepare_data.py
python src\train_tokenizer.py
python src\train.py
```

### 5. Перевод
```cmd
python src\infer.py --text "Բարև, աշխարհ" --checkpoint checkpoints\best.pt
```

## Обучение на Kaggle GPU (рекомендуется)

Если локального GPU нет, загрузите `kaggle_train.ipynb` на Kaggle:
1. kaggle.com → Notebooks → New Notebook
2. Загрузите файл `kaggle_train.ipynb`
3. Включите GPU (T4 x2)
4. Запустите все ячейки
5. Скачайте чекпоинты из панели Output → скопируйте на `D:\armenian_nlp\checkpoints\`

## Реалистичные ожидания

Это мини-версия. С 1000–5000 парами получится прототип, который переводит простые фразы. Для качественного перевода нужно 50 000+ параллельных предложений.

## Ресурсы Kaggle (используются в проекте)

- `ffatty/plaintext-wikipedia-full-armenian` — современный армянский текст
- `alvations/old-newspapers` — старые газеты (грабар)
- `arthurarstamyan/1500-correctly-punctuated-armsents` — семенные предложения
- `atamazian/gemma-2-armenian-alpaca-lora` — армянская языковая модель (для генерации синтетических пар)
