# Как работает код — объяснение логики

Разбор реализации из `week_1_pure_python/`. Весь NLP написан с нуля, без внешних библиотек.

---

## Общий поток данных

```
Текстовый файл
    ↓ tokenizer.py
Список предложений, каждое — список слов
    ↓ model.py (NgramLM.fit)
Словарь + таблица счётчиков n-грамм
    ↓ model.py (score / logscore / generate)
Вероятности слов / сгенерированные токены
    ↓ generate.py (generate_sent)
Читаемый текст
    ↓ serialize.py
Файл .pkl на диске
```

---

## Шаг 1 — Токенизация (`tokenizer.py`)

Задача: превратить сырой текст в список списков слов.

### Разбивка на предложения

```python
re.split(r"(?<=[.!?])\s+", text)
```

`(?<=[.!?])` — это *lookbehind*: находим позицию сразу после точки/восклицательного/вопросительного знака. `\s+` — один или несколько пробелов после. Результат: текст разбивается по границам предложений.

### Разбивка на слова

```python
re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
```

Паттерн ищет два варианта (через `|`):
- `\w+(?:'\w+)*` — слово, возможно с апострофом внутри (`don't`, `it's`)
- `[^\w\s]` — одиночный знак препинания (`.`, `,`, `!` и т.д.) как отдельный токен

Пример:
```
"Language is never, ever random."
→ ['Language', 'is', 'never', ',', 'ever', 'random', '.']
```

### Два режима токенизации

- `tokenize_text(text)` — для связного текста (статья): сначала разбивает на предложения, потом каждое на слова + переводит в нижний регистр.
- `tokenize_sentences(sentences)` — для уже разбитых строк (твиты): каждую строку токенизирует как одно предложение, без поиска границ.

---

## Шаг 2 — Обучение модели (`model.py`)

### Что такое n-граммная модель

N-граммная модель предсказывает следующее слово, опираясь на n−1 предыдущих. Для триграммной модели (n=3): `P(слово | два_предыдущих_слова)`.

Вместо нейронной сети — просто подсчёт частот в корпусе.

### Специальные токены

Каждое предложение оборачивается в специальные маркеры:
- `<s>` — начало предложения (BOS, Beginning Of Sentence)
- `</s>` — конец предложения (EOS, End Of Sentence)
- `<UNK>` — любое слово, не встреченное при обучении

Для триграмм перед каждым предложением добавляется **два** `<s>` (n−1 штук), чтобы у первых слов тоже был контекст:

```
["the", "cat", "sat"]
→ ["<s>", "<s>", "the", "cat", "sat", "</s>"]
```

### Подсчёт n-грамм

Для каждого предложения модель считает **все** k-граммы от 1 до n:

```python
for k in range(1, self.n + 1):
    for i in range(len(padded) - k + 1):
        kgram = tuple(padded[i : i + k])
        context = kgram[:-1]   # всё кроме последнего слова
        word = kgram[-1]       # последнее слово
        self._counts[k][context][word] += 1
```

Структура хранения (`_counts`):
```
_counts[порядок][контекст_как_кортеж][слово] = количество
```

Пример для `["<s>", "<s>", "the", "cat", "sat", "</s>"]`:

| Порядок | Контекст | Слово | Счётчик |
|---------|----------|-------|---------|
| 1 (унигр.) | `()` | `"the"` | +1 |
| 2 (бигр.) | `("<s>",)` | `"the"` | +1 |
| 3 (тригр.) | `("<s>", "<s>")` | `"the"` | +1 |
| 3 (тригр.) | `("<s>", "the")` | `"cat"` | +1 |
| 3 (тригр.) | `("the", "cat")` | `"sat"` | +1 |

---

## Шаг 3 — Вычисление вероятностей (`model.score`)

```python
def score(self, word, context=None) -> float:
    ctx = tuple(context[-(self.n - 1):])   # берём последние n-1 слов
    order = len(ctx) + 1
    ctx_total = sum(self._counts[order][ctx].values())
    return self._counts[order][ctx][word] / ctx_total
```

Формула: **P(слово | контекст) = count(контекст + слово) / count(контекст)**

Пример:
```
P("never" | ["language", "is"])
= count("language", "is", "never") / count("language", "is")
= 7 / 11
≈ 0.636
```

`logscore` — то же самое, но в логарифмической шкале (`log2`). Нужен чтобы избежать численного переполнения при перемножении многих маленьких вероятностей.

---

## Шаг 4 — Генерация текста

### Как работает `model.generate`

```python
def generate(self, num_words, random_seed=42):
    rng = random.Random(random_seed)     # воспроизводимый генератор
    context = ["<s>", "<s>"]            # начальный контекст для триграмм
    result = []

    for _ in range(num_words):
        ctx = tuple(context[-(self.n - 1):])
        candidates = self._counts[self.n][ctx]   # все слова, встречавшиеся после этого контекста
        token = rng.choices(
            list(candidates.keys()),
            weights=list(candidates.values())    # выбираем пропорционально частоте
        )[0]
        result.append(token)
        if token == "</s>":
            break
        context.append(token)

    return result
```

`rng.choices(слова, weights=частоты)` — взвешенная случайная выборка. Слово с частотой 10 выбирается в 10 раз чаще, чем слово с частотой 1.

`random_seed` фиксирует результат: один и тот же seed → одно и то же предложение.

### Как работает `generate_sent`

`generate()` возвращает сырые токены включая `<s>` и `</s>`. `generate_sent()` убирает их и детокенизирует:

```python
for token in model.generate(num_words, random_seed):
    if token == "<s>":
        continue      # пропустить маркер начала
    if token == "</s>":
        break         # остановиться на маркере конца
    content.append(token)
return _detokenize(content)
```

### Детокенизация

```python
def _detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r" ([?.!,;:'\")\]])", r"\1", text)   # убрать пробел перед знаками
    text = re.sub(r"\s'(s|t|re|ve|ll|d|m)\b", r"'\1", text)  # починить сокращения
    return text
```

Превращает `["language", "is", ",", "never"]` в `"language is, never"`.

---

## Шаг 5 — Сериализация (`serialize.py`)

Модель сохраняется на диск через stdlib `pickle`:

```python
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

`NgramLM` содержит только обычные Python-объекты (`dict`, `Counter`, `set`) — поэтому стандартный pickle справляется. В NLTK-варианте (`week_1_python`) пришлось использовать `dill`, потому что внутри NLTK MLE есть lambda-функции, которые обычный pickle не умеет сериализовать.

---

## Структура пакета

```
ngram_lm/
├── __main__.py    argparse CLI — точка входа, склеивает модули
├── data.py        загрузка текста и CSV (только I/O, никакой NLP-логики)
├── tokenizer.py   regex-токенизация текста и предложений
├── model.py       класс NgramLM: fit / score / logscore / generate
├── generate.py    generate_sent() — фильтрация токенов + детокенизация
└── serialize.py   save_model / load_model через pickle
```

Каждый модуль отвечает за одну задачу. `__main__.py` — единственное место, где они соединяются вместе.
