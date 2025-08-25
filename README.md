# practicum-workshop-cvd-model

Максимально простое приложение для предсказания отношения к группе риска сердечного приступа высокий/низкий.

Первая итерация - отсутствует веб-интерфейс, требуется много доработок и рефакторинга, модель так же требует тюнинга(очень слабый recall), но минимально-необходимый функционал - есть.
> Метрики:  
> accuracy: 75.198%  
> recall: 29.134%  
> precision: 90.244%  
> ROC-AUC: 0.74  
  
## Требования

- Python 3.9

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Запуск сервиса

Из корня репозитория:

```bash
python practicum-workshop-cvd-model/practicum-workshop-cvd-model.py \
  --model-path models/model.joblib \
  --preprocessor-path models/preprocessor.joblib \
  --host 0.0.0.0 \
  --port 8000
```

Проверка статуса:

```bash
curl -s http://localhost:8000/health
```

Интерактивная документация будет доступна на `http://localhost:8000/docs`.

## Эндпоинты

- `GET /health` — проверка сервиса.
- `POST /predict_path` — предсказание по CSV сервиса - принимает путь до файла на сервере, ответ приходит в виде JSON.
- `POST /predict_save_path` — предсказание по CSV с сохранением результата в CSV файл.

### Пример запроса

`/predict_path` - базовый запрос, ответ в виде JSON:

```bash
curl -s -X POST http://localhost:8000/predict_path \
  -H 'Content-Type: application/json' \
  -d '{"csv_path": "datasets/heart_test.csv"}'
```

Сохранение в CSV:

```bash
curl -s -X POST http://localhost:8000/predict_save_path \
  -H 'Content-Type: application/json' \
  -d '{"csv_path": "datasets/heart_test.csv", "output_csv_path": "predictions.csv"}'
```

## Формат входных данных

Ожидается, что входные признаки соответствуют тем, на которых обучалась модель и препроцессор.
