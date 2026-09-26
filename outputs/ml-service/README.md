# ML-сервис прогноза для backend

Реализует ML Prediction Port backend (Runoi/LCT-backend 928b993, src/services/ml_port.py): `POST /predict` и `GET /health`. Модель: HGB run-002 из ml-baseline-v2/decision.json; при старте сверяется SHA-256 модели.

```bash
.venv/bin/python outputs/ml-service/service.py \
  --prepared work/ml-prepared --config outputs/ml-dataset/dataset-config.json \
  --decision outputs/ml-baseline-v2/decision.json --port 8090
```

Старт около 3 секунд, ответ 13 мс по медиане и 87 мс в 95-м перцентиле на 340 запросах. Stdlib http.server, без внешних зависимостей сверх requirements baseline.

## Контракт

Вход: JSON backend (target_type, target_id, risk_type, as_of, recent_alarm_count, recent_anomaly_count, window_hours). Используются только target_id и as_of: признаки считаются по журналу канала за 30 суток до as_of, счётчики backend игнорируются, потому что относятся к пачке событий, а не к фиксированному окну. target_id принимается как `sensor_<ид_канала>` или как ид канала. as_of без часового пояса считается UTC.

Ответ 200: поля PredictionResult (probability, lead_min_hours = 0, prediction_window_hours = 24, top_factors, recommendation, model_name) плюс status, alert, threshold, as_of_utc, target_scope = recorded_alarm_episode_start и features. top_factors перечисляют входные значения, это не вклад признаков в модель.

probability означает ожидаемую частоту начала записанного тревожного эпизода канала в следующие 24 часа. Это не вероятность аварии. На test 2026 года в бинах score 0,4–0,7 частота завышена (см. ml-baseline-v2/README.md).

Ответ 422 `{"status": "insufficient_data", "reason": ...}` вместо выдуманного числа:

| reason | условие |
|---|---|
| global_history | журнал начинается позже чем за 30 суток до as_of |
| stale_journal | as_of позже последней записи журнала больше чем на час |
| unknown_channel | у канала нет записей до as_of |
| no_recent_records | у канала нет записей за 30 суток |
| channel_history | канал наблюдается меньше суток |
| known_log_gap | общая пауза журнала в последние 24 часа |
| active_alarm | тревога за последние 15 минут: эпизод продолжается, модель прогнозирует только новые |

400 при неверном JSON, дате, пустом target_id или теле больше 64 KB.

## Прогноз по объекту: инцидент (run-008) и отказ (run-009)

```bash
B=outputs/ml-baseline-v2
.venv/bin/python outputs/ml-service/service.py --prepared work/ml-prepared --config outputs/ml-dataset/dataset-config.json \
  --decision $B/decision.json --directory <справочник_каналов_датчиков.csv> \
  --object-decision $B/run-008-incident72/object-decision.json --object-decision $B/run-009-neispraven168/object-decision.json \
  --port 8090 --demo-anchor 2026-06-20T00:00:00
```

Без `--object-decision` сервис работает как раньше. SHA-256 моделей и калибровок сверяются при старте, состояния класса берутся из object-decision.json. Зависимости: outputs/ml-service/requirements.txt (добавлен shap). Старт около 10 секунд.

1. `POST /predict_object {"object_id": "3215", "as_of": "...", "target": "incident"}`: прогноз по одному объекту. target необязателен, по умолчанию incident.
2. `POST /risk_map {"as_of": "...", "target": "failure"}`: все 78 объектов справочника, отсортированы по вероятности, затем по score; поле alerts с числом объектов выше порога. Около секунды.

| target | что прогнозируется | горизонт | test 2026: precision / recall | базовая доля | правило: precision / recall |
|---|---|---|---|---|---|
| incident | начало эпизода инцидента в объекте («Не замкнут», «Обнаружен дым», «Затоплен» и др.) | 72 ч | 0,827 / 0,616 | 0,347 | 0,573 / 0,492 |
| failure | начало эпизода «Неисправен» в объекте | 168 ч | 0,869 / 0,625 | 0,239 | 0,343 / 0,428 |

Поля ответа: target, target_scope, horizon_hours, probability (isotonic на 2025 году; на test у incident корзины отклоняются от наблюдаемой доли не больше чем на 5 п. п., у failure модель занижает до 13 п. п.; отчёты в run-008-calibration и run-009-calibration), score и threshold модели, alert = score ≥ threshold (порогу соответствует probability 0,652 у incident и 0,559 у failure), top_factors (до трёх признаков с наибольшим положительным вкладом SHAP, текстом), suspect_channels (до пяти каналов объекта с наибольшей оценкой канальной модели run-002 на 24 часа: вторая ступень, где искать), channels_used и channels_total. 400 при неизвестном target, 404 unknown_object, 422 при общей нехватке данных (stale_journal, known_log_gap, global_history) и no_eligible_channels.

Цель та же, что в обучении: начало записанного эпизода, не подтверждённая авария. Каналы вне справочника в прогноз объекта не входят.

Проверки 26.09: признаки объекта сервиса совпали с обучающими на 600 случайных объекто-сутках 2019–2025 годов (сиды 7 и 23), включая историю инцидентов, 0 расхождений (`check_object_parity.py`, выход 1 при расхождении); пакетный расчёт признаков каналов совпал с поканальным на 1 600 парах канал-дата; 20 HTTP-проверок на реальных данных прошли после одной правки (при равной вероятности isotonic рейтинг теперь идёт по score). После добавления failure: признаки совпали на 300 объекто-сутках для failure (сид 7) и ещё 300 для incident (сид 31), 24 HTTP-проверки с обеими моделями прошли. Тесты: test_feature_store_many, test_object_predictor, test_object_predictor_end_to_end (синтетический журнал, маленькие настоящие модели), test_service_objects; мутационная проверка по новому коду сервиса: из 17 мутантов первым прогоном пойманы 15, два выживших (фильтр класса в class_first, передача цели в тексты причин) пойманы после доработки тестов.

## Проверки

1. Признаки сервиса на реальных данных совпали с training-examples-v2 на 300 случайных пригодных снимках 2019–2026 годов, 0 расхождений; причины отказа совпали на 40 снимках active_alarm и channel_history.
   Повторяемая сверка (26.09): `check_parity.py <work/ml-prepared> <ml-dataset/dataset-config.json> <training-examples-v2.parquet> [seed]`, выход 1 при любом расхождении. Слои: пригодные снимки, снимки с эпизодами, тревогами за сутки, паузами журнала, исключённые и active_alarm. На сидах 7, 11 и 23 обслужено 1 035 снимков, расхождений 0; из исключённых сервис обслуживает только те, чья причина исключения лежит в будущем (split_boundary, future_window, будущая пауза журнала). Сверка ловит сдвиг окна 7 суток, иной порог эпизода и сдвиг окна пауз.
2. 20 HTTP-проверок на реальных данных: 9 успешных прогнозов (включая as_of с поясом +03:00 и оба формата target_id), 4 причины 422, 4 варианта 400, health, 404. probability сервиса совпала с офлайн-score модели на 8 снимках test с точностью 1e-12.

## Стык с backend

### Риски по объектам: backend-object-risks.patch (26.09)

Патч к Runoi/LCT-backend 928b993, накладывается отдельно и поверх backend-risk-sync.patch (`git apply --check` на чистом клоне). Новый src/services/object_risk_sync.py: при старте backend, если задан ML_PREDICTOR_URL, запрашивает `/risk_map` для incident и failure и создаёт Risk с target_type="facility", target_id=fac_<ид_объект> для каждого объекта с alert. Тип риска: для failure sensor_failure; для incident по первому каналу-подозреваемому (пожарные датчики → fire, охранные → unauthorized_access, насосы и затопление → flooding, иначе unauthorized_access). Уровень риска считает их risk_level_for_probability по калиброванной probability. Повтор не создаёт дубль, пока у объекта есть открытый риск той же модели с незакончившимся окном. Ошибка ML-сервиса не роняет старт (ObjectRiskError в лог). Тесты в их каталоге: tests/test_object_risk_sync.py, 4 теста, ML-сервис подменён httpx.MockTransport.

Проверено 26.09 на Postgres 16 в Docker: полный набор их тестов на чистой базе 208 passed без патча и 212 passed с патчем. На повторном прогоне по той же базе у них падают 75 тестов и без нашего патча, наши 4 теста повтор переживают. Сквозной запуск backend (оба патча) с нашим сервисом на реальном датасете: в /risks под manager 27 рисков по объектам, ровно 13 предупреждений incident и 14 failure из /risk_map; dispatcher видит только объекты своей зоны. Линтер в их репозитории не настроен (нет ruff/flake8 в конфиге и requirements), поэтому линтером не проверялось; код следует их форматированию, без docstrings.

### Прогноз по каналам

Две проблемы найдены при чтении backend 928b993 и подтверждены запуском:

1. Backend передаёт as_of = datetime.now(UTC), а архив журнала заканчивается 30 июня 2026 года. Без демо-часов сервис на текущее время отвечает stale_journal.
2. risk_sync.py не перехватывает MLPredictorError, а первый sync_risks идёт в lifespan. Контрольный запуск без патча: `Application startup failed` на первом ответе 422 (no_recent_records).

Решение для демо:

1. Сервис с флагом `--demo-anchor 2026-06-20T00:00:00`: запрошенное время сдвигается на постоянную величину (якорь минус момент старта сервиса). Ответ содержит demo_clock с исходным as_of и якорем, /health показывает demo_now_utc. Признаки считаются только по журналу до сдвинутого времени. Запас демо около 10 суток до конца журнала. Без флага режим строгий.
2. Патч backend-risk-sync.patch для команды backend: канал пропускается при MLPredictorError, курсор событий всё равно сдвигается (`state.last_synced_event_id = rows[-1].id`), зацикливания нет. Патч применяется к 928b993 через `git apply`. В репозиторий backend он не вносился.

Покрытие каналов примера backend (1 241 канал) при якоре 2026-06-20: 1 194 получают прогноз, 47 без записей за 30 суток получают 422.

Сквозная проверка 25.09: Postgres 16 в Docker, копия backend 928b993 с патчем, `ML_PREDICTOR_URL` на сервис с якорем 2026-06-20. Backend стартовал; в таблице risks 134 прогноза с model = hgb-v2-run002-c2c25f5d, probability 0,0006–0,875, уровни: low 130, medium 3, critical 1. 44 из 581 события остались без прогноза из-за 422. Тесты backend (pytest) не запускались.

Запуск для демо:

```bash
.venv/bin/python outputs/ml-service/service.py --prepared work/ml-prepared \
  --config outputs/ml-dataset/dataset-config.json --decision outputs/ml-baseline-v2/decision.json \
  --port 8090 --demo-anchor 2026-06-20T00:00:00
# backend: ML_PREDICTOR_URL=http://127.0.0.1:8090 и применённый backend-risk-sync.patch
```

Пороги уровней риска backend (0,3 / 0,6 / 0,85) не совпадают с рабочим порогом модели 0,147. Большинство каналов попадут в «Низкий», даже когда alert = true. Согласовать шкалу с командой backend: либо уровни по порогу модели, либо показывать поле alert.
