# Ранжирование коротких видео

Полный ML-пайплайн для задачи ранжирования ленты коротких
видео на датасете взаимодействий пользователей с видео (в используемой
выборке — ~48 млн взаимодействий, 10 тыс. пользователей, ~20 тыс. видео,
27 недель с глобальным временным порядком событий).

Стек: **Polars / PyArrow** (data engineering), **PyTorch** (эмбеддинг-фичи,
cuda/mps), **CatBoost** (YetiRank-ранкер, обучение на GPU), **ClearML**
(трекинг экспериментов, локальный сервер в Docker).

## Структура

```
configs/
  pipeline.yaml        # конфиг data-этапов (индексация, очистка, фичи, сплиты)
  train.yaml           # конфиг итоговой модели CatBoost
src/
  common.py            # конфиги, ClearML, выбор torch-устройства
  prepare_data.py      # этап 1: индексация данных, manifest.yaml
  preprocess.py        # этап 2: очистка + градуированная релевантность (label)
  features.py          # этап 3: юзерские / айтемные / юзер-айтемные фичи
  build_dataset.py     # этап 4: темпоральный сплит train/val/test
  train.py             # этап 5: CatBoost YetiRank на GPU + оффлайн-метрики
  metrics.py           # NDCG/MAP/MRR/HitRate/Recall/Precision@k, GAUC
  pipeline.py          # оркестратор: запускает этапы последовательно
tests/
  conftest.py          # синтетический мини-датасет + фикстуры прогона этапов
  test_metrics.py      # метрики: ручные расчеты, сверка со sklearn
  test_prepare_data.py # индексация данных, временной порядок шардов
  test_preprocess.py   # очистка, label, фильтры активности
  test_features.py     # агрегаты, train-окно (без утечек), emb_cos
  test_build_dataset.py# сплиты, группы CatBoost, отсутствие утечек таргета
  test_train.py        # smoke-обучение + модель против случайного ранжирования
docker-compose.yml     # локальный ClearML server (Mac arm64 -> linux/amd64)
```

## 1. Данные

Исходные данные лежат в `data/raw/` (путь настраивается в
`configs/pipeline.yaml` → `paths.raw_dir`):

```
data/raw/
  **/week_NN.parquet                 # недельные шарды взаимодействий
  metadata/users_metadata.parquet    # пользователи: age, gender, geo
  metadata/items_metadata.parquet    # видео: duration, author_id
  metadata/item_embeddings.npz       # контентные эмбеддинги видео
```

Схема взаимодействий: `user_id`, `item_id`, `timespent`, фидбек
(`like, dislike, share, bookmark, click_on_author, open_comments`) и
контекст показа (`place, platform, agent`). Датасет отсортирован по
времени — это используется для темпорального сплита.

## 2. Окружение

CatBoost не собирается под Python 3.14 — нужен **Python 3.9–3.12**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Локальный ClearML server

`docker-compose.yml` адаптирован под Apple Silicon: у образов ClearML нет
arm64-варианта, поэтому всем сервисам ClearML прописана
`platform: linux/amd64` (Docker эмулирует).

```bash
mkdir -p ~/clearml-data/mongo ~/clearml-data/mongo/configdb \
         ~/clearml-data/fileserver ~/clearml-data/logs ~/clearml-data/config \
         ~/clearml-data/agent ~/clearml-data/elastic ~/clearml-data/redis
docker compose up -d
docker ps        # проверить, что все контейнеры поднялись
```

Веб-интерфейс: <http://localhost:8080> (логин по умолчанию `admin` / `admin`).
API — порт 8008, файловый сервер — 8081.

Подключение SDK: в веб-интерфейсе Settings → Workspace → Create new
credentials, затем в активированном venv:

```bash
clearml-init   # вставить скопированный блок с ключами
```

Появится `~/clearml.conf` с `api_server: http://localhost:8008`. Если сервер
недоступен или в конфиге `clearml.enabled: false`, пайплайн работает без
трекинга (только логи в консоль).

## 4. Запуск пайплайна

```bash
python -m src.pipeline                 # все этапы: prepare -> ... -> train
python -m src.pipeline --stages train  # только обучение
```

Каждый этап — отдельный эксперимент в проекте `ShortVideo-Ranking` в ClearML
(конфиги, статистики, кривые обучения, метрики, артефакты).

Для быстрой локальной проверки в `configs/pipeline.yaml`:

```yaml
data_source:
  max_shards: 2          # только первые 2 недельных шарда
sample:
  user_fraction: 0.1     # и 10% пользователей из них
```

и в `configs/train.yaml`:

```yaml
data:
  max_rows: 1000000      # первые 1M строк каждого сплита
```

## 5. Что делает каждый этап

### Этап 1 — prepare (индексация данных)
Сканирует `data/raw/`, раскладывает файлы по категориям (взаимодействия /
пользователи / видео / эмбеддинги) и сортирует недельные шарды **по номеру
недели** (сортировка по пути ломает временной порядок). Результат —
`manifest.yaml`, который читают следующие этапы.

### Этап 2 — preprocess (очистка + label)
- отбрасываются события с пустыми id, отрицательным/аномальным `timespent`,
  нулевой длительностью; `watch_ratio = timespent / duration` клипается;
- фильтр минимальной активности (>= 5 событий на пользователя и на видео);
- **градуированная релевантность** для ранжирования:
  `label = 1·[watch_ratio >= 0.8] + 2·like + 3·share + 3·bookmark +
  1·click_on_author + 1·open_comments`, дизлайк обнуляет, клип до 5.
  Веса — в `configs/pipeline.yaml`.

Все на Polars в streaming-режиме — объемы в память не загружаются.

### Этап 3 — features
Все агрегаты считаются **только на train-окне** (первые 80% событий по
времени) — защита от утечки будущего в валидацию.

| Группа | Фичи |
|---|---|
| Юзерские | age, gender, geo; активность, средний watch_ratio/timespent, доли like/dislike/share/..., число уникальных авторов, средняя длительность просматриваемого |
| Айтемные | duration; популярность, средний watch_ratio, средний label, доли фидбека; агрегаты автора (охват, число роликов, доли фидбека) |
| Юзер-айтемные | история юзера с автором ролика (`ua_*`); `dur_diff` = \|duration − привычная юзеру длительность\|; `emb_cos` — косинус контентного эмбеддинга видео (первые 32 компоненты, L2-нормировка) и профиля юзера (среднее эмбеддингов его train-позитивов), считается на torch (cuda/mps/cpu) чанками |
| Контекст показа | place, platform, agent (известны на момент ранжирования) |

Фидбек текущего события (like, timespent и т.д.) в фичи не входит — утечка.

### Этап 4 — build_dataset
Темпоральный сплит 80/10/10 по глобальному порядку событий, джойн фичей,
сортировка по (user_id, event_order) — группы CatBoost должны быть
непрерывными. Списки фичей сохраняются в `dataset_meta.yaml`.

### Этап 5 — train (итоговая модель)
`CatBoostRanker` с лоссом **YetiRank**, группы = пользователи, `task_type:
GPU` (на машине без CUDA автоматически падает на CPU —
`allow_cpu_fallback`). Все гиперпараметры — в `configs/train.yaml`.

Метрики **на этапе обучения**: CatBoost по итерациям считает на валидации
`NDCG@10, MAP@10, MRR@10, PrecisionAt@10, RecallAt@10` (секция
`custom_metric`), кривые уезжают в ClearML.

После обучения на val и test считается полный набор **собственных**
оффлайн-метрик из `src/metrics.py`.

## 6. Оффлайн-метрики (src/metrics.py)

Все метрики групповые (группа = пользователь), усредняются по группам;
реализация векторизована (numpy lexsort + reduceat), без циклов по группам.

- **NDCG@k** — качество порядка с учетом градуированной релевантности
  (гейн `2^rel − 1`); основная метрика ранжирования.
- **MAP@k** — средняя точность по позициям позитивов в топ-k.
- **MRR@k** — обратная позиция первого позитива.
- **HitRate@k** — есть ли хоть один позитив в топ-k.
- **Recall@k / Precision@k** — полнота/точность топ-k.
- **GAUC** — средний по пользователям ROC-AUC (Манн-Уитни).

Позитив для бинарных метрик: `label >= 1` (порог в `configs/train.yaml`).
Группы без позитивов исключаются из усреднения.

## 7. Тестирование пайплайна и модели

```bash
python -m pytest tests/ -v
```

Тесты не требуют ни GPU, ни реальных данных: `tests/conftest.py` генерирует
синтетический мини-датасет (та же структура: недельные шарды, метаданные,
npz-эмбеддинги) с **подсаженным сигналом** — у каждого видео есть латентное
качество, от которого зависят и досмотры, и лайки. Этапы пайплайна
прогоняются по цепочке фикстур один раз за сессию.

| Файл | Что проверяет |
|---|---|
| `test_metrics.py` | NDCG/MAP/MRR/HitRate/Recall/Precision/GAUC: ручные расчеты, сверка NDCG со sklearn, исключение групп без позитивов |
| `test_prepare_data.py` | раскладка файлов по категориям, временной порядок недельных шардов, manifest |
| `test_preprocess.py` | фильтры аномалий (timespent, watch_ratio), min-активность, джойн метаданных, монотонность event_order, точное совпадение label с формулой конфига, обнуление по дизлайку |
| `test_features.py` | агрегаты сходятся с пересчетом руками; **считаются только по train-окну** (нет утечки будущего); emb_cos в [-1, 1] и по строке на событие |
| `test_build_dataset.py` | сплиты без пересечений и по границам времени, непрерывность групп для CatBoost, **отсутствие таргетных колонок в фичах**, типы/NaN, наличие всех трех групп фичей |
| `test_train.py` | smoke-обучение CatBoost (CPU, 60 итераций): артефакты на месте, метрики в [0, 1], **модель обыгрывает случайное ранжирование** (NDCG@10, GAUC > 0.55), feature importance покрывает все фичи |

## 8. Артефакты обучения

- `models/catboost_ranker.cbm` — модель;
- `models/offline_metrics.json` — метрики val/test;
- `models/feature_importance.csv` — важности фичей;
- все это же — артефакты ClearML-таска + кривые обучения и конфиги.
