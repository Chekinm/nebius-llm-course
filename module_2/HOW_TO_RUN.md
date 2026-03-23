# Как запустить — все три варианта

В папке `module_2/` три параллельных реализации одного и того же проекта:

| Папка | Что это | Зависимости |
|-------|---------|-------------|
| `week_1/` | Jupyter-ноутбук | nltk, dill, pandas, ipykernel |
| `week_1_python/` | Python-пакет с NLTK | nltk, dill, pandas |
| `week_1_pure_python/` | Python-пакет без внешних NLP-либ | pandas, requests |

---

## Вариант 1 — Jupyter Notebook (`week_1/`)

```bash
cd module_2/week_1

# Установить зависимости
uv sync

# Скачать ресурсы NLTK (один раз)
uv run python -c "import nltk; nltk.download('punkt_tab')"

# Запустить Jupyter
uv run jupyter notebook
```

Открыть файл `Ngram_Language_Model_with_NLTK.ipynb` и выполнять ячейки сверху вниз (`Shift+Enter`).

---

## Вариант 2 — Python + NLTK (`week_1_python/`)

```bash
cd module_2/week_1_python

# Установить зависимости
uv sync

# Скачать ресурсы NLTK (один раз)
uv run python -c "import nltk; nltk.download('punkt_tab')"
```

### Команды

```bash
# Обучить модель на статье Килгаррифа (сохранится в models/)
uv run python -m ngram_lm train kilgarriff

# Обучить модель на твитах Трампа
uv run python -m ngram_lm train trump

# Сгенерировать текст из модели
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl

# Сгенерировать 3 предложения с seed=7, до 30 слов
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl --count 3 --seed 7 --words 30

# Оценить вероятность слова
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl language

# P(never | language is)
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl never --context language is

# Полный демо-запуск (обе модели, несколько предложений)
uv run python -m ngram_lm demo
```

---

## Вариант 3 — Pure Python (`week_1_pure_python/`)

```bash
cd module_2/week_1_pure_python

# Установить зависимости (никакого NLTK и dill нет)
uv sync
```

Команды — **идентичны варианту 2**:

```bash
uv run python -m ngram_lm train kilgarriff
uv run python -m ngram_lm train trump
uv run python -m ngram_lm generate models/kilgarriff_ngram_model.pkl --count 3 --seed 7
uv run python -m ngram_lm score models/kilgarriff_ngram_model.pkl never --context language is
uv run python -m ngram_lm demo
```

Никакого скачивания NLTK ресурсов не нужно — просто `uv sync` и запуск.

---

## Справка по аргументам CLI

```
train <corpus> [--n N] [--output PATH]
    corpus     kilgarriff  — статья Килгаррифа
               trump       — твиты Трампа
    --n        порядок n-грамм (по умолчанию 3)
    --output   куда сохранить .pkl (по умолчанию models/<corpus>_ngram_model.pkl)

generate <MODEL_PATH> [--words W] [--seed S] [--count C]
    --words    максимум токенов в предложении (по умолчанию 20)
    --seed     начальный random seed (по умолчанию 42)
    --count    сколько предложений сгенерировать (семена: seed, seed+1, ...)

score <MODEL_PATH> <word> [--context word1 word2 ...]
    word       слово для оценки
    --context  контекст (предшествующие слова)

demo           запустить полный pipeline без аргументов
```

---

## Где хранятся данные

Оба Python-варианта (`week_1_python` и `week_1_pure_python`) читают данные напрямую из `week_1/` — дублирования нет:

```
module_2/
  week_1/
    language-never-random.txt   ← используется всеми вариантами
    Donald-Tweets!.csv          ← используется всеми вариантами
  week_1_python/
  week_1_pure_python/
```

Если файл `language-never-random.txt` отсутствует, он скачается автоматически при первом запуске `train kilgarriff`.
