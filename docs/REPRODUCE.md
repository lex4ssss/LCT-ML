# Воспроизведение

Тесты проверяют код без реального датасета. Чтобы обучить модели или запустить сервис, нужен журнал из архива организаторов, собранный по шагам ниже.

## Окружение и тесты

Нужен Python 3.12: в более старых версиях нет hashlib.file_digest.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r outputs/ml-service/requirements.txt
ML_PYTHON=.venv/bin/python bash scripts/test.sh
```

Тесты работают на синтетических входах, без сети и реального датасета; тесты baseline выполняют небольшое искусственное обучение. Наборы запускаются отдельными процессами, потому что у ранних тестовых модулей совпадают имена.

## Данные

Все подготовленные данные лежат в work/ml-prepared, этот каталог в git не входит. Годовые наблюдения называются 2019.parquet–2026.parquet, эпизоды тревог episodes-900.parquet, выборка снимков training-examples-v2.parquet. Файл с суффиксом calendar-only является отклонённой версией, использовать его нельзя.

## Сборка из архива организаторов

Возьми датасет, перечисленный в [sources.json](sources.json), и сверь SHA-256. Пароль архивов в репозитории не хранится; годовые 7z читаются через bsdtar. Для полного прохода нужен bsdtar с поддержкой этих 7z и место для распаковки одного года, кэшей и временных файлов. Скрипт проверяет нижний порог свободного места, но это не обещание достаточности места для всей процедуры.

```bash
mkdir -p work/rebuilt-audit work/ml-prepared work/rebuilt-preparation
.venv/bin/python outputs/ml-archive/profile_archives.py \
  /path/to/archive work/archive-audit work/rebuilt-audit
```

Режим: полный строгий проход, без очистки, обучения и инференса. Каталог отчётов должен быть новым; кэш не должен содержать годовых результатов предыдущего запуска. Агрегированные отчёты из outputs не перезаписываются.

Для восьми полученных годовых Parquet вызови существующую функцию prepare. Дедупликация включена для 2019 и 2023 годов, где полные повторы подтверждены аудитом. Если исходник отличается от зафиксированного датасета, сначала пересмотри это решение по результатам нового аудита.

```bash
PYTHONPATH=outputs/ml-dataset .venv/bin/python - <<'PY'
import json
from pathlib import Path
from prepare import prepare

files = sorted(Path('work/archive-audit').glob('20??.parquet'))
assert len(files) == 8
for source in files:
    result = prepare(source,Path('work/ml-prepared')/source.name,source.stem in {'2019','2023'})
    (Path('work/rebuilt-preparation')/(source.stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
PY
```

Затем примени проверенный SQL эпизодов и сверь общие паузы с конфигурацией. Здесь обучение и инференс выключены, gap=900 секунд, порог паузы=3600 секунд. Выход эпизодов должен отсутствовать.

```bash
PYTHONPATH=outputs/ml-dataset .venv/bin/python - <<'PY'
import json
from pathlib import Path
import duckdb
from prepare import literal

root = Path('work/ml-prepared')
target = root/'episodes-900.parquet'
assert not target.exists()
files = sorted(root.glob('20??.parquet'))
assert len(files) == 8
with duckdb.connect(config={'threads':1,'memory_limit':'1GB','temp_directory':'work/rebuild-spill'}) as connection:
    connection.execute('CREATE VIEW observations AS SELECT * FROM read_parquet(['+','.join(map(literal,files))+'])')
    connection.execute('SET VARIABLE gap_seconds = 900')
    query = Path('outputs/ml-dataset/episodes.sql').read_text()
    connection.execute('COPY ('+query+') TO '+literal(target)+' (FORMAT PARQUET, COMPRESSION ZSTD)')
    connection.execute('SET VARIABLE gap_threshold_seconds = 3600')
    actual = connection.execute(Path('outputs/ml-dataset/log_gaps.sql').read_text()).fetchall()
    expected = json.loads(Path('outputs/ml-dataset/dataset-config.json').read_text())['gap_intervals_utc']
    assert [(str(a),str(b)) for a,b,_ in actual] == [(g['start_at'],g['end_at']) for g in expected]
PY
```

Если проверка интервалов не проходит, не продолжай обучение: проверь версии исходника, очистку и часовой пояс. Не подгоняй конфигурацию под ожидаемые метрики. Фрагмент собран из существующих компонентов; целиком одной командой полный проход по архиву не повторялся.

Собери снимки текущей постановки:

```bash
.venv/bin/python outputs/ml-dataset/build_dataset.py \
  work/ml-prepared work/ml-prepared/training-examples.parquet \
  outputs/ml-dataset/dataset-config.json > work/rebuilt-dataset-run.json
```

CLI требует совпадения gap с соседним episodes-900-run.json и не перезаписывает существующий Parquet. Метки и признаки должны совпасть по смыслу и агрегатам с outputs/ml-dataset/dataset-run.json. Байтовый хеш заново записанного Parquet не обязан совпасть при изменении версии библиотеки или параметров записи.

## Повторение baseline

Режим: обучение только train, порог по validation; исходная конфигурация суточных окон и gap=900; class_weight, калибровка, оценка test и онлайн-инференс выключены. Используется новый каталог, исходный run-001 сохраняется.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  outputs/ml-baseline/fit_baseline.py \
  work/ml-prepared/training-examples.parquet work/baseline-reproduced

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  outputs/ml-baseline/inspect_validation.py \
  work/ml-prepared/training-examples.parquet \
  work/ml-prepared/episodes-900.parquet work/baseline-reproduced
```

Сохранённый validation-diagnostics.json дополнен разовой независимой проверкой порога и рассчитанными долями покрытия. inspect_validation.py воспроизводит основные счётчики, интервалы калибровки, покрытие и хеши, но не добавляет поля independent_threshold_sweep, total_episodes_in_validation_windows, episode_coverage_fraction и episodes_in_alerted_windows_fraction. Эти доли выводятся из его счётчиков; проверка порога описана в REVIEW.md. Не трактуй различие набора полей как изменение качества модели.

## Выборка v2, модели объектов и сервис

Выборка v2 и эпизоды по классам: [outputs/ml-dataset-v2/README.md](../outputs/ml-dataset-v2/README.md). Эксперименты run-002…run-009 и команды их запуска: [outputs/ml-baseline-v2/README.md](../outputs/ml-baseline-v2/README.md). Сервис: [outputs/ml-service/README.md](../outputs/ml-service/README.md).

## Backend

Патчи в outputs/ml-service написаны против Runoi/LCT-backend на коммите 928b993; данные backend в этот репозиторий не копируются.

```bash
git clone https://github.com/Runoi/LCT-backend.git work/LCT-backend
git -C work/LCT-backend checkout 928b9931da16ca53498425316482d19bc40f11ba
```
