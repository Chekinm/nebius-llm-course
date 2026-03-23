# N-gram Language Models — Pure Python (без внешних NLP-библиотек)

Третий вариант реализации N-граммных языковых моделей. Никаких NLTK, dill или других NLP-зависимостей — только Python stdlib и `pandas`/`requests` для загрузки данных.

| Вариант | Папка | Описание |
|---------|-------|----------|
| Jupyter notebook | `week_1/` | Интерактивный ноутбук с объяснениями |
| Python + NLTK | `week_1_python/` | CLI-пакет на базе NLTK MLE |
| Pure Python | `week_1_pure_python/` | CLI-пакет, только stdlib |

## Структура проекта

```
week_1_pure_python/
├── pyproject.toml          зависимости: только pandas + requests
├── .python-version         Python 3.14
├── models/                 сюда сохраняются обученные модели (.pkl)
└── ngram_lm/
    ├── __main__.py         CLI: train / generate / score / demo
    ├── data.py             загрузка данных из ../week_1/
    ├── tokenizer.py        regex-токенизатор (без NLTK punkt)
    ├── model.py            класс NgramLM — своя реализация MLE
    ├── generate.py         генерация текста + regex-детокенизатор
    └── serialize.py        сохранение через stdlib pickle (без dill)
```

## Установка

```bash
cd module_2/week_1_pure_python
uv sync
```

Никаких дополнительных шагов не нужно — внешние NLP-ресурсы не используются.

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

### Генерация текста

```bash
# Сгенерировать одно предложение
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl

# Сгенерировать 3 предложения с seed 7, до 30 слов каждое
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl --count 3 --seed 7 --words 30

# Генерация из модели твитов
uv run python -m ngram_lm generate models/trump_ngram_model.pkl --words 40 --count 5
```

### Оценка вероятности слова

```bash
# Вероятность слова без контекста (унигрaммная)
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl language

# Условная вероятность: P(never | language is)
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl never --context language is
```

Вывод: вероятность и логарифмическая вероятность (log2):
```
P('never' | language is) = 0.636364  [log2 = -0.6521]
```

### Полная демонстрация

```bash
uv run python -m ngram_lm demo
```

Обучает обе модели и генерирует примеры — весь pipeline за одну команду.

## Как устроена реализация

### Класс `NgramLM` (`model.py`)

Хранит счётчики n-грамм всех порядков от 1 до n в структуре:
```
_counts[order][context_tuple][word] = frequency
```

- **`fit()`** — формирует словарь, затем для каждого предложения добавляет паддинг `<s>`/`</s>` и считает все k-граммы (k от 1 до n)
- **`score(word, context)`** — P(word|context) = count(context+word) / count(context)
- **`generate()`** — семплирует токены по одному из условного распределения с заданным `random.Random(seed)`

### Токенизатор (`tokenizer.py`)

Два regex-паттерна вместо NLTK punkt:
- Предложения: `re.split(r'(?<=[.!?])\s+', text)`
- Слова: `re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)`

### Сериализация (`serialize.py`)

Используется stdlib `pickle` — `NgramLM` не содержит lambda-функций, поэтому стандартного pickle достаточно (в отличие от NLTK MLE, где требовался `dill`).

## Данные

Файлы данных берутся из `../week_1/`:
- `language-never-random.txt` — статья Килгаррифа (скачается автоматически если отсутствует)
- `Donald-Tweets!.csv` — датасет твитов Дональда Трампа
