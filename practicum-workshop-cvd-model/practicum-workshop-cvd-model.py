import argparse
import io
import re
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# TODO: выенести это все отсюда в отдельный файл
def format_column_name(col_name: str) -> str:
    """
    Метод форматирования строк.

    Args:
        col_name (_type_): име колонки

    Returns:
        str: отформатированное имя колонки
    """
    name = col_name.lower()
    # TODO удаляет г из слов, потом исправить
    name = re.sub(r"\s*(%|,|\.|\(|\))\s*", " ", name)
    name = re.sub(r"-", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")


def format_column_names(data: pd.DataFrame):
    """
    Метод для переименования столбцов во всем датафрейме.

    Args:
        data (pd.DataFrame): датафрейм, в котором надо отформатировать названяи столбцов.
    """
    data.columns = [format_column_name(col) for col in data.columns]


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_strategy="median",
        cat_strategy="most_frequent",
        replace_strings=None,
        replace_value=np.nan,
    ):
        """
        Кастомный инпутер для обработки значений в данных.

        Args:
            num_strategy (str, optional): стратегия для числовых признаков.
            Defaults to 'median'.
            cat_strategy (str, optional): стратегия для категориальных признаков.
            Defaults to 'most_frequent'.
            replace_strings (_type_, optional): список строк для замены на NaN
            (например, [' ', '?', 'unknown']). Defaults to None.
            replace_value (_type_, optional): значение, на которое заменяем
            указанные строки. Defaults to np.nan.
        """
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.replace_strings = replace_strings
        self.replace_value = replace_value
        self.num_imputer = SimpleImputer(strategy=num_strategy)
        self.cat_imputer = SimpleImputer(strategy=cat_strategy)

    def fit(self, X, y=None):
        X = X.copy()
        if self.replace_strings is not None:
            X.replace(self.replace_strings, self.replace_value, inplace=True)
        self.num_cols = X.select_dtypes(include=np.number).columns
        self.cat_cols = X.select_dtypes(exclude=np.number).columns
        if not self.num_cols.empty:
            self.num_imputer.fit(X[self.num_cols])
        if not self.cat_cols.empty:
            self.cat_imputer.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self.replace_strings is not None:
            X.replace(self.replace_strings, self.replace_value, inplace=True)
        if not self.num_cols.empty:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])
        if not self.cat_cols.empty:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])
        return X


class CvdRiskModelService:
    """Сервис загрузки артефактов и выполнения предсказаний.

    Инкапсулирует препроцессор и модель. Предполагается, что препроцессор
    совместим со sklearn API (имеет метод transform), а модель — метод predict.
    """

    def __init__(self, model_path: str, preprocessor_path: str) -> None:
        self.model_path: str = model_path
        self.preprocessor_path: str = preprocessor_path
        self._model: Any = None
        self._preprocessor: Any = None

    def load_artifacts(self) -> None:
        try:
            self._model = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить модель: {self.model_path}"
            ) from exc

        try:
            self._preprocessor = joblib.load(self.preprocessor_path)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить препроцессор: {self.preprocessor_path}"
            ) from exc

    def _ensure_ready(self) -> None:
        if self._model is None or self._preprocessor is None:
            raise RuntimeError(
                "Артефакты не загружены. Вызовите load_artifacts()."
            )

    def predict_dataframe(self, data_frame: pd.DataFrame) -> List[Any]:
        """Возвращает список предсказаний для переданного DataFrame.

        Возвращает значения из model.predict(...) как список python-типов.
        """
        self._ensure_ready()

        try:
            transformed = self._preprocessor.transform(data_frame)
        except Exception as exc:
            raise ValueError(
                "Ошибка трансформации данных препроцессором"
            ) from exc

        try:
            raw_predictions = self._model.predict(transformed)
        except Exception as exc:
            raise ValueError("Ошибка вычисления предсказаний моделью") from exc

        if isinstance(raw_predictions, (list, tuple)):
            result = list(raw_predictions)
        elif isinstance(raw_predictions, np.ndarray):
            result = raw_predictions.tolist()
        else:
            # Единичное значение — приведём к списку
            result = [raw_predictions]

        return result

    def predict_records(self, records: List[Dict[str, Any]]) -> List[Any]:
        """Прогноз для списка записей в формате словарей."""
        frame = pd.DataFrame.from_records(records)
        return self.predict_dataframe(frame)


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Any]
    num_instances: int
    id_column: Optional[str] = None


class PredictPathRequest(BaseModel):
    csv_path: str


class PredictSavePathRequest(BaseModel):
    csv_path: str
    output_csv_path: str


def create_app(service: CvdRiskModelService) -> FastAPI:
    app = FastAPI(title="CVD Risk Prediction API", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict_path", response_model=PredictResponse)
    def predict_path(payload: PredictPathRequest) -> PredictResponse:
        try:
            df = pd.read_csv(payload.csv_path)
            # TODO: вынести в функцию
            df["Gender"] = df["Gender"].replace(
                {"1.0": "Male", "0.0": "Female"}
            )
            df.drop(columns=["Unnamed: 0"], inplace=True)
            df = df.set_index("id")
            format_column_names(df)
            df = df.dropna()
            df = df.drop(["diet"], axis=1)
            predictions = service.predict_dataframe(df)
            return PredictResponse(
                predictions=predictions, num_instances=len(df)
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict_save_path")
    def predict_save_path(payload: PredictSavePathRequest) -> Dict[str, Any]:
        """Считывает CSV, предсказывает и сохраняет результат в CSV с колонками id,prediction."""
        try:
            df = pd.read_csv(payload.csv_path)
            # Повторяем ту же подготовку, что и в predict_path
            df["Gender"] = df["Gender"].replace(
                {"1.0": "Male", "0.0": "Female"}
            )
            df = df.dropna()

            df.drop(columns=["Unnamed: 0"], inplace=True)
            # Сохраним id до изменения индекса
            if "id" in df.columns:
                ids = df["id"].copy()
                df = df.set_index("id")
            else:
                # Если id уже индекс — используем его
                ids = df.index.to_series().rename("id")
            format_column_names(df)
            df = df.dropna()
            if "diet" in df.columns:
                df = df.drop(["diet"], axis=1)

            predictions = service.predict_dataframe(df)
            out_df = pd.DataFrame(
                {"id": ids.values, "prediction": predictions}
            )
            out_df.to_csv(payload.output_csv_path, index=False)
            return {
                "status": "ok",
                "saved_to": payload.output_csv_path,
                "num_instances": int(len(out_df)),
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVD Risk Prediction Service")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Путь к сохранённой модели",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default="models/preprocessor.joblib",
        help="Путь к сохранённому препроцессору",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для запуска сервера",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для запуска сервера",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = CvdRiskModelService(
        model_path=args.model_path, preprocessor_path=args.preprocessor_path
    )
    service.load_artifacts()

    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
