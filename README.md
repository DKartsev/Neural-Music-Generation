# Chopin Composer — Neural Music Generation Research Project

- Русская версия: [Перейти к русской версии](#русская-версия)

## Overview
This project presents a research-grade system for generating musical sequences in the style of **Frédéric Chopin**, implemented as a reproducible Colab pipeline. It models expressive, continuous-time piano music using three atomic event attributes:

- **Pitch** (MIDI note number)
- **Step** (time since previous note, in seconds)
- **Duration** (note length, in seconds)

The pipeline trains a dual-input recurrent neural network on MIDI transcriptions attributed to Chopin and generates new sequences conditioned on seed material from Mozart. The focus is on capturing temporal expressivity (rubato, microtiming) and melodic-harmonic tendencies present in Chopin’s piano writing.

## Quick facts (code-accurate)
- `seq_length` (context window): **32** time steps
- Pitch embedding dimension: **64**
- LSTM layers: **256 (return_sequences=True) → 128** units
- Time projection: `TimeDistributed(Dense(64, activation='relu'))`
- Loss weights: `{'pitch_out': 1.0, 'time_out': 10.0}`
- Default training epochs: **30**; adaptive batch size **128** or **32**
- Dataset URL: `https://storage.yandexcloud.net/academy.ai/classical-music-midi.zip`
- Configurable dataset limits: `MAX_CHOPIN_FILES = 200`, `MAX_SEQS = 20000`

---

## Architecture (ASCII diagram — exact structure from code)

```
Inputs:
  - pitch_input: (batch, seq_length)        # integer pitch indices
  - numeric_input: (batch, seq_length, 2)   # normalized (step, duration)

pitch_input --> Embedding( input_dim=vocab_size, output_dim=64 ) -->
                                                       \
                                                        +--> Concatenate --> LSTM(256, return_sequences=True) --> LSTM(128) -->
numeric_input --> TimeDistributed(Dense(64, relu)) --> /                                                     |                     
                                                                                                              |                     
                                                                                                     /--------+--------\           
                                                                                                    v                 v          
                                                                                         Dense(vocab_size, softmax)   Dense(2, linear)  
                                                                                         (pitch_out)                (time_out: step_norm,dur_norm)
```

Notes:
- Both dropout and recurrent_dropout are set to 0.2 on LSTM layers in the code.
- The numeric branch projects (step,duration) per timestep to 64 dims to align with the pitch embedding before concatenation.

---

## Data discovery & preprocessing (exactly as implemented)
1. The notebook downloads and unpacks the dataset to `classical_midis/`.
2. It discovers all `.mid` and `.midi` files recursively.
3. Composer selection: files are selected by filename substring (`'chopin'`, `'mozart'`) and by inspecting `pretty_midi` instrument `name` fields if present. A path-substring fallback is applied when initial detection returns no files.
4. For each selected Chopin file the code:
   - prefers piano instruments (name contains 'piano' or MIDI program 0–7)
   - extracts `(pitch, start, end)` for every note and sorts by onset
   - converts to `(pitch, step, duration)` where `step` is **0.0 for the first event** in each file and thereafter `max(0.0, start - prev_start)`; `duration = max(1e-4, end - start)`
5. Global statistics `max_step` and `max_dur` are computed across the Chopin corpus and used for normalization; default fallback values are 1.0 if these are zero.
6. A **dense pitch vocabulary** is constructed covering the integer range `min_pitch..max_pitch` observed in the Chopin corpus (inclusive). This design ensures a contiguous mapping from MIDI pitch numbers to indices.
7. Sequence construction: sliding windows of length `seq_length=32` are created per-file; the next event after each window serves as the supervised target.
8. Time features are normalized by dividing by `max_step` and `max_dur` and clipped to `[0.0, 1.0]` (durations lower-clipped to `1e-6`).

Implementation details (code variables you will find in the notebook):
- `all_events` — raw list of (pitch, step, duration) collected from Chopin files
- `pitch_to_idx`, `idx_to_pitch` — mappings saved to `pitch_map.pkl`
- `X_pitches`, `X_numeric` — training arrays

---

## Training (code-accurate protocol)
- Targets: joint optimization of categorical next-pitch (`y_pitch_cat`) and continuous time (`y_time` with shape (N,2)).
- Train/validation split: `train_test_split(..., test_size=0.1, random_state=42)`.
- Checkpointing: `ModelCheckpoint` saves the best weights to `best_chopin_model.h5` monitoring `val_loss`.
- The model is compiled with optimizer `adam`, loss `{'pitch_out':'categorical_crossentropy','time_out':'mse'}` and loss weights `{'pitch_out':1.0,'time_out':10.0}`.

Practical notes:
- Loss scaling emphasizes time regression to offset differing magnitudes between cross-entropy and MSE on normalized time values.
- No explicit EarlyStopping callback is used in the current script (only best-checkpoint saving).

---

## Generation (exact behavior implemented)
- **Seed extraction**: `build_seed_from_mozart` shuffles available Mozart files and extracts the first `seq_length` usable events. If a Mozart seed is not available, a random Chopin training sequence is used and padded if needed.
- **Pitch mapping**: each seed pitch is mapped to the dense Chopin vocabulary. If a pitch lies outside the `min_pitch..max_pitch` range, it is clamped to the nearest boundary value.
- **Sampling**: temperature is applied **only to the pitch softmax** via `sample_with_temperature`. If `temperature <= 0.0`, argmax is used.
- **Time decoding**: the model predicts normalized `step` and `duration` which are then clipped to `[0.0,1.0]`, small Gaussian noise is added (scale = `0.01 * max(0.5, temperature)`), and values are denormalized by `max_step` and `max_dur`.
- **Output**: sequences are converted to MIDI with absolute-onset accumulation (`current_time += step`) and saved as `.mid`. The notebook attempts to render `.wav` using `timidity` after MIDI export.

Default example generation in the script: `generated_seq = generate_music(model, seed, length=300, temperature=1.0)` and variants for `temp in [0.5,1.0,1.5]` are also created and saved.

---

## Visualization & Analysis (code snippets you can run in Colab)
Below are the exact code snippets (copy‑paste ready) used to produce the visualizations included in this repository. Run these **after** preprocessing (i.e., after `all_events` is populated).

**Pitch distribution**
```python
import matplotlib.pyplot as plt
pitches = [e[0] for e in all_events]
plt.figure(figsize=(10,4))
plt.hist(pitches, bins=range(min(pitches), max(pitches)+2))
plt.title('Pitch Distribution (Chopin corpus)')
plt.xlabel('MIDI pitch')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('pitch_distribution.png', dpi=200)
plt.show()
```

**Step (inter-onset interval) distribution**
```python
steps = [e[1] for e in all_events]
plt.figure(figsize=(10,4))
plt.hist(steps, bins=100)
plt.title('Step Distribution (Inter-onset intervals)')
plt.xlabel('Step (seconds)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.grid(True)
plt.savefig('step_distribution.png', dpi=200)
plt.show()
```

**Duration distribution**
```python
durations = [e[2] for e in all_events]
plt.figure(figsize=(10,4))
plt.hist(durations, bins=100)
plt.title('Duration Distribution (Note lengths)')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.grid(True)
plt.savefig('duration_distribution.png', dpi=200)
plt.show()
```

**Piano-roll plotting** (already included in the notebook as `plot_pianoroll(sequence)`)

---

## Artifacts & reproducibility
Files produced by the code (names are exact):
- `best_chopin_model.h5`
- `pitch_map.pkl` (contains `pitch_to_idx`, `idx_to_pitch`, `min_pitch`, `max_pitch`, `max_step`, `max_dur`)
- `generated_chopin_like.mid`, `generated_chopin_like.wav` (if rendering succeeds)
- `generated_temp_0.5.mid/.wav`, `generated_temp_1.0.mid/.wav`, `generated_temp_1.5.mid/.wav`

To reproduce: run the Colab notebook end‑to‑end. The only non-deterministic steps are sampling with temperature and injected Gaussian noise; setting Python/NumPy/TensorFlow seeds can reduce variance across runs.

---

## Limitations & Future Work
- The model encodes polyphonic events into a single event stream and thus does not explicitly model separate voices; future work may explore multi-voice encodings or event-based transformer models.
- MIDI transcription quality varies across sources; manual curation or composer verification would strengthen corpus validity.
- Time regression is continuous and noisy; optional quantization to a musically-relevant grid (e.g., 16th-note subdivisions) could improve rhythmic coherence.
- Additional conditioning (harmony, dynamics, phrase boundaries) would improve stylistic control and musical form.

---

## References & Related Work (suggested reading)
- Magenta project (Google Brain) — MusicVAE, Melody RNN
- "BachBot" style symbolic generation research
- Recent transformer-based symbolic music models (e.g., Music Transformer)


---


---

## Русская версия

# Chopin Composer — Нейросетевая генерация музыки (исследовательский проект)

## Обзор
Этот проект представляет исследовательскую систему для генерации музыкальных последовательностей в стиле **Фредерика Шопена**, реализованную как воспроизводимый конвейер для Colab. Модель описывает экспрессивную, непрерывно-временную фортепианную музыку с использованием трёх атомарных атрибутов события:

- **Pitch** (высота тона, номер MIDI)
- **Step** (время с момента предыдущей ноты, в секундах)
- **Duration** (длительность ноты, в секундах)

Конвейер обучает рекуррентную нейросеть с двойным входом на MIDI‑транскрипциях, приписываемых Шопену, и генерирует новые последовательности, условно заданные исходным материалом из Моцарта. Акцент сделан на захвате темповой экспрессии (рубато, микро‑тайминг) и мелодико‑гармонических тенденций, присущих фортепианной манере Шопена.

## Краткие факты
- `seq_length` (окно контекста): **32** временных шага
- Размер эмбеддинга высоты: **64**
- LSTM слои: **256 (return_sequences=True) → 128** юнитов
- Проекция времени: `TimeDistributed(Dense(64, activation='relu'))`
- Веса потерь: `{'pitch_out': 1.0, 'time_out': 10.0}`
- Эпох по умолчанию: **30**; адаптивный размер батча **128** или **32**
- URL датасета: `https://storage.yandexcloud.net/academy.ai/classical-music-midi.zip`
- Настраиваемые ограничения датасета: `MAX_CHOPIN_FILES = 200`, `MAX_SEQS = 20000`

---

## Архитектура (ASCII-диаграмма)

```
Inputs:
  - pitch_input: (batch, seq_length)        # integer pitch indices
  - numeric_input: (batch, seq_length, 2)   # normalized (step, duration)

pitch_input --> Embedding( input_dim=vocab_size, output_dim=64 ) -->
                                                       \
                                                        +--> Concatenate --> LSTM(256, return_sequences=True) --> LSTM(128) -->
numeric_input --> TimeDistributed(Dense(64, relu)) --> /                                                     |                     
                                                                                                              |                     
                                                                                                     /--------+--------\           
                                                                                                    v                 v          
                                                                                         Dense(vocab_size, softmax)   Dense(2, linear)  
                                                                                         (pitch_out)                (time_out: step_norm,dur_norm)
```

Примечания:
- В коде для LSTM-слоёв заданы `dropout` и `recurrent_dropout` равные 0.2.
- Числовая ветвь (время) проецирует `(step,duration)` для каждого шага в пространство размерности 64, чтобы сопоставиться с эмбеддингом высоты до конкатенации.

---

## Обнаружение данных и предобработка
1. Ноутбук скачивает и распаковывает датасет в `classical_midis/`.
2. Он рекурсивно находит все файлы с расширениями `.mid` и `.midi`.
3. Выбор композитора: файлы отбираются по подстроке имени файла (`'chopin'`, `'mozart'`) и по полю `name` инструмента из `pretty_midi`, если оно присутствует. Если первичный поиск не даёт результатов, применяется резервный отбор по подстроке пути.
4. Для каждого выбранного файла Шопена код:
   - отдаёт предпочтение фортепианным инструментам (имя содержит 'piano' или MIDI-программа в диапазоне 0–7)
   - извлекает `(pitch, start, end)` для каждой ноты и сортирует по моменту атаки
   - преобразует в `(pitch, step, duration)`, где `step` равен **0.0 для первого события** в файле, а далее `max(0.0, start - prev_start)`; `duration = max(1e-4, end - start)`
5. По корпусу Шопена вычисляются глобальные статистики `max_step` и `max_dur`, используемые для нормализации; при нулевых значениях по умолчанию берётся 1.0.
6. Строится **плотный словарь высот** (dense pitch vocabulary) покрывающий целочисленный диапазон `min_pitch..max_pitch`, обнаруженный в корпусе Шопена (включительно). Это обеспечивает непрерывное отображение номеров MIDI в индексы.
7. Конструирование последовательностей: для каждого файла создаются скользящие окна длины `seq_length=32`; следующее событие после окна служит целью обучения.
8. Временные признаки нормализуются делением на `max_step` и `max_dur` и обрезаются в диапазоне `[0.0, 1.0]` (длительности дополнительно нижне‑ограничиваются `1e-6`).

Реализационные переменные, которые вы найдёте в ноутбуке:
- `all_events` — необработанный список кортежей `(pitch, step, duration)`, собранных из файлов Шопена
- `pitch_to_idx`, `idx_to_pitch` — отображения, сохраняемые в `pitch_map.pkl`
- `X_pitches`, `X_numeric` — массивы обучения

---

## Обучение
- Цели: совместная оптимизация категориальной следующей высоты `y_pitch_cat` и непрерывного времени `y_time` формы `(N,2)`.
- Разделение на обучение/валидацию: `train_test_split(..., test_size=0.1, random_state=42)`.
- Контрольные точки: `ModelCheckpoint` сохраняет лучшие веса в `best_chopin_model.h5`, мониторя `val_loss`.
- Модель компилируется с оптимизатором `adam`, loss `{'pitch_out':'categorical_crossentropy','time_out':'mse'}` и весами потерь `{'pitch_out':1.0,'time_out':10.0}`.

Практические замечания:
- Масштабирование потерь отдаёт приоритет регрессии времени, чтобы компенсировать различие в масштабах между кросс‑энтропией и MSE для нормализованных временных величин.
- В текущем скрипте явного `EarlyStopping` нет (только сохранение лучшей контрольной точки).

---

## Генерация
- **Формирование посевного сегмента (seed)**: `build_seed_from_mozart` перемешивает доступные файлы Моцарта и извлекает первые `seq_length` пригодных событий. Если посев из Моцарта недоступен, используется случайная обучающая последовательность из Шопена с возможным дополнением до нужной длины.
- **Преобразование высот**: каждая высота семени отображается на плотный словарь Шопена. Если высота выходит за пределы `min_pitch..max_pitch`, она зажимается до ближайшей границы.
- **Сэмплинг**: температура применяется **только к softmax высот** через `sample_with_temperature`. Если `temperature <= 0.0`, используется `argmax`.
- **Декодирование времени**: модель предсказывает нормализованные `step` и `duration`, затем значения обрезаются до `[0.0,1.0]`, добавляется небольшой гауссов шум (масштаб = `0.01 * max(0.5, temperature)`), и они денормализуются умножением на `max_step` и `max_dur`.
- **Выход**: последовательности конвертируются в MIDI с накоплением абсолютного времени (`current_time += step`) и сохраняются как `.mid`. Ноутбук пытается отрендерить `.wav` через `timidity` после экспорта MIDI.

Пример генерации по умолчанию в скрипте: `generated_seq = generate_music(model, seed, length=300, temperature=1.0)`; также создаются варианты для `temp in [0.5,1.0,1.5]` и сохраняются.

---

## Визуализация и анализ
Ниже — точные фрагменты кода (копировать и вставлять), использованные для визуализаций. Запускайте их **после** предобработки (т.е. после заполнения `all_events`).

**Распределение высот**
```python
import matplotlib.pyplot as plt
pitches = [e[0] for e in all_events]
plt.figure(figsize=(10,4))
plt.hist(pitches, bins=range(min(pitches), max(pitches)+2))
plt.title('Pitch Distribution (Chopin corpus)')
plt.xlabel('MIDI pitch')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('pitch_distribution.png', dpi=200)
plt.show()
```

**Распределение интервалов (step)**
```python
steps = [e[1] for e in all_events]
plt.figure(figsize=(10,4))
plt.hist(steps, bins=100)
plt.title('Step Distribution (Inter-onset intervals)')
plt.xlabel('Step (seconds)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.grid(True)
plt.savefig('step_distribution.png', dpi=200)
plt.show()
```

**Распределение длительностей**
```python
durations = [e[2] for e in all_events]
plt.figure(figsize=(10,4))
plt.hist(durations, bins=100)
plt.title('Duration Distribution (Note lengths)')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.grid(True)
plt.savefig('duration_distribution.png', dpi=200)
plt.show()
```

**Пианоролл** (функция уже включена в ноутбук как `plot_pianoroll(sequence)`)

---

## Артефакты и воспроизводимость
Файлы, которые генерируются кодом:
- `best_chopin_model.h5`
- `pitch_map.pkl` (содержит `pitch_to_idx`, `idx_to_pitch`, `min_pitch`, `max_pitch`, `max_step`, `max_dur`)
- `generated_chopin_like.mid`, `generated_chopin_like.wav` (если рендеринг прошёл успешно)
- `generated_temp_0.5.mid/.wav`, `generated_temp_1.0.mid/.wav`, `generated_temp_1.5.mid/.wav`

Для воспроизведения: запустите ноутбук Colab от начала до конца. Единственные недетерминированные шаги — сэмплинг с температурой и добавление Гауссова шума; установка зерен для Python/NumPy/TensorFlow может снизить разброс результатов между запусками.

---

## Ограничения и дальнейшая работа
- Модель кодирует полифонию как единый поток событий и, следовательно, не моделирует явно отдельные голоса; в будущем можно исследовать кодировки с несколькими голосами или event-based трансформеры.
- Качество MIDI‑транскрипций варьируется по источникам; ручная кураторская проверка или верификация композитора повысили бы надёжность корпуса.
- Регрессия времени даёт непрерывные и шумные предсказания; опциональная квантизация к музыкально релевантной сетке (например, 16‑е ноты) может улучшить ритмическую когерентность.
- Дополнительная условность (гармония, динамика, границы фраз) даст лучший контроль над стилем и формой музыкальных результатов.

---

## Литература и связанные работы (рекомендации)
- Проект Magenta (Google Brain) — MusicVAE, Melody RNN
- Исследования в стиле "BachBot" по символической генерации
- Современные модели на базе трансформеров для символической музыки (например, Music Transformer)

---
