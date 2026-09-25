# ML Москоллектора


1. Аудит полного архива и подготовка выборки завершены для прогноза записанного тревожного эпизода.
2. Первый baseline обучен, проверен на validation и сохранён вместе с отчётами. Качество и покрытие пока недостаточны для внедрения.
3. Test не использовался для оценки модели. Калибровка и интеграция с backend остаются открытыми.

## Где лежит результат

1. [outputs/ml-workplan.md](outputs/ml-workplan.md): постановка, декомпозиция и расхождения с backend.
2. [outputs/ml-archive](outputs/ml-archive): полный аудит источника; [outputs/ml-dataset](outputs/ml-dataset): подготовка, признаки, метки и проверки.
3. [outputs/ml-baseline](outputs/ml-baseline): обучение, тесты, модель run-001 и подробное критическое ревью.


## Быстрая проверка


```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r outputs/ml-baseline/requirements.txt
ML_PYTHON=.venv/bin/python bash scripts/test.sh
```


Режим сохранённого baseline: история и горизонт по 24 часа, gap эпизода 900 секунд, маска общей паузы от 3600 секунд, очистка включена; обучение train и выбор порога validation выполнены; калибровка, оценка test и онлайн-инференс выключены. Precision на validation 21,38%, recall 38,43%; пригодные снимки покрывают 40,75% эпизодов. Это не показатели прогноза физических поломок.
