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

## Проверки

1. Признаки сервиса на реальных данных совпали с training-examples-v2 на 300 случайных пригодных снимках 2019–2026 годов, 0 расхождений; причины отказа совпали на 40 снимках active_alarm и channel_history.
2. 20 HTTP-проверок на реальных данных: 9 успешных прогнозов (включая as_of с поясом +03:00 и оба формата target_id), 4 причины 422, 4 варианта 400, health, 404. probability сервиса совпала с офлайн-score модели на 8 снимках test с точностью 1e-12.

## Стык с backend

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
