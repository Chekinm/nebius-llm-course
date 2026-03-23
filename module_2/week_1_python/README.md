# N-gram Language Models — Pure Python

Переписка Jupyter-ноутбука `week_1/Ngram_Language_Model_with_NLTK.ipynb` в виде чистого Python-пакета с CLI-интерфейсом. Реализует обучение MLE n-граммных моделей, генерацию текста и оценку вероятностей через командную строку.

## Структура проекта

```
week_1_python/
├── pyproject.toml          # конфигурация uv-проекта и зависимости
├── .python-version         # Python 3.14
├── models/                 # сюда сохраняются обученные модели (.pkl)
└── ngram_lm/
    ├── __main__.py         # CLI: train / generate / score / demo
    ├── data.py             # загрузка данных (текст Килгаррифа и твиты Трампа)
    ├── tokenizer.py        # токенизация с fallback на ToktokTokenizer
    ├── model.py            # обучение MLE-модели
    ├── generate.py         # генерация текста
    └── serialize.py        # сохранение и загрузка модели через dill
```

## Установка

```bash
cd module_2/week_1_python
uv sync
```

## Загрузка ресурсов NLTK

Один раз после установки:

```bash
uv run python -c "import nltk; nltk.download('punkt_tab')"
```

> Если `punkt_tab` недоступен, пакет автоматически переключится на запасной токенизатор `ToktokTokenizer` — всё продолжит работать.

## Использование

### Обучение модели

```bash
# Обучить 3-граммную модель на тексте статьи Килгаррифа
uv run python -m ngram_lm train kilgarriff

# Обучить модель на твитах Трампа
uv run python -m ngram_lm train trump

# Задать другой порядок n-грамм или путь сохранения
uv run python -m ngram_lm train kilgarriff --n 4 --output models/my_model.pkl
```

Модели сохраняются в `models/kilgarriff_ngram_model.pkl` и `models/trump_ngram_model.pkl`.

### Генерация текста

```bash
# Сгенерировать одно предложение (seed=42, до 20 слов)
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl

# Сгенерировать 3 предложения с разными seed, до 30 слов каждое
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl --count 3 --seed 7 --words 30

# Генерация из модели на твитах
uv run python -m ngram_lm generate models/trump_ngram_model.pkl --words 40 --count 5
```

### Оценка вероятности слова

```bash
# Вероятность слова без контекста (унигррамная)
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl language

# Условная вероятность: P(never | language is)
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl never --context language is
```

Вывод включает вероятность и логарифмическую вероятность (log2):
```
P('never' | language is) = 0.636364  [log2 = -0.6521]
```

### Полная демонстрация

Запускает весь pipeline сразу: обучает обе модели и генерирует примеры текста.

```bash
uv run python -m ngram_lm demo
```

## Как работает пакет

```
data.py          →  tokenizer.py  →  model.py   →  serialize.py
(загрузка текста)   (токенизация)    (обучение)     (сохранение)
                                         ↓
                                    generate.py
                                   (генерация текста)
```

`__main__.py` — точка входа, объединяет все модули через argparse.

## Данные

Файлы данных берутся напрямую из соседней папки `../week_1/`:
- `language-never-random.txt` — статья Адама Килгаррифа (если файла нет, скачается автоматически)
- `Donald-Tweets!.csv` — датасет твитов Дональда Трампа

Дублировать файлы не нужно.

## Почему `dill` вместо `pickle`

Стандартный `pickle` не может сериализовать `nltk.lm.MLE`-модели, так как внутри они содержат лямбда-функции. Библиотека `dill` расширяет возможности pickle и решает эту проблему прозрачно.
