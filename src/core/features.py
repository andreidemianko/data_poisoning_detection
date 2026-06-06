"""
Векторизация датасета для статистических методов обнаружения отравления.

Числовые колонки — медиана + StandardScaler.
Категориальные — OrdinalEncoder с заменой пропусков.
Текст — TF-IDF по главной текстовой колонке (макс. 200 признаков).

Результат кладётся в context.metadata["features"] сканером FeatureVectorizerScanner,
который должен запуститься раньше всех остальных в категории stats.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

_TEXT_AVG_LEN_THRESHOLD = 25      # средняя длина строки выше этого порога → текстовая колонка
_CAT_MAX_CARDINALITY = 50         # больше уникальных значений → категориальная колонка не нужна
_TFIDF_MAX_FEATURES = 200         # максимум TF-IDF признаков


@dataclass
class FeatureBundle:
    """
    Матрица признаков и сопутствующие метаданные после векторизации.

    X             — float64-массив (n_samples × n_features), без NaN/inf
    y             — вектор меток или None
    label_col     — название целевой колонки в исходном датафрейме
    feature_names — названия признаков (совпадает с колонками X)
    dataset_type  — "tabular" / "text" / "mixed"
    meta          — n_numeric, n_categorical, n_text_tfidf, text_column и т.д.
    """

    X: np.ndarray
    y: Optional[np.ndarray]
    label_col: Optional[str]
    feature_names: List[str]
    dataset_type: str
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── convenience properties ─────────────────────────────────────────────────
    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def classes(self) -> Optional[np.ndarray]:
        return np.unique(self.y) if self.y is not None else None

    def X_for_class(self, cls: Any) -> np.ndarray:
        """Строки X, соответствующие метке cls."""
        if self.y is None:
            raise ValueError("No labels available in this FeatureBundle.")
        return self.X[self.y == cls]

    def X_tabular(self) -> np.ndarray:
        """Только числовые + категориальные признаки, без TF-IDF. Для M1-M7."""
        n_tab = self.meta.get("n_numeric", 0) + self.meta.get("n_categorical", 0)
        return self.X[:, :n_tab] if n_tab > 0 else self.X

    def tabular_feature_names(self) -> List[str]:
        n_tab = self.meta.get("n_numeric", 0) + self.meta.get("n_categorical", 0)
        return self.feature_names[:n_tab] if n_tab > 0 else self.feature_names


def reduce_to_n_components(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Снижает размерность X через PCA до n_components.
    Нужно для методов, требующих p << n (Mahalanobis/MinCovDet, LOF, DBSCAN).
    Векторизатор намеренно не делает PCA, чтобы M1-M7 и M14 видели все признаки.
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    if n_components <= 0 or n_components >= X.shape[1]:
        return X
    return PCA(n_components=n_components, random_state=42).fit_transform(X)


# точные совпадения (чувствительные к регистру) — самые распространённые имена целевой переменной
_LABEL_EXACT = (
    "label", "target", "y", "class", "Class",
    "SeriousDlqin2yrs",  # специфично для датасета give_me_some_credit
)

# ключевые слова для нечёткого поиска (нормализованное имя колонки должно совпадать или начинаться с одного из них)
_LABEL_KEYWORDS = frozenset({
    "label", "target", "class", "y",
    "exited", "churn", "default", "fraud", "risk",
    "toxic", "spam", "sentiment",
})


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Ищет целевую колонку в три уровня:
      1 — точное совпадение с общепринятыми именами
      2 — нормализованное совпадение с ключевыми словами (Exited→exited, Risk→risk)
      3 — имя начинается с ключевого слова (default.payment.next.month → default)
    """
    for c in _LABEL_EXACT:
        if c in df.columns:
            return c

    for col in df.columns:
        norm = re.sub(r"[^a-z]", "", col.lower())
        if norm in _LABEL_KEYWORDS:
            return col
        for kw in _LABEL_KEYWORDS:
            if norm.startswith(kw) and len(norm) > len(kw):
                return col

    return None


def _is_string_col(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _classify_columns(
    df: pd.DataFrame, label_col: Optional[str]
) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Разделяет колонки датафрейма на три группы: числовые, категориальные, текстовая.
    Текстовая — одна (с наибольшей средней длиной строки), остальные длинные текстовые игнорируются.
    """
    exclude = {label_col} if label_col else set()
    num_cols: List[str] = []
    cat_cols: List[str] = []
    text_candidates: List[Tuple[float, str]] = []  # (avg_len, col_name)

    for col in df.columns:
        if col in exclude:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col].dtype):
            num_cols.append(col)
            continue

        if _is_string_col(df[col]):
            try:
                avg_len = float(series.astype(str).str.len().mean())
            except Exception:
                continue
            if avg_len >= _TEXT_AVG_LEN_THRESHOLD:
                text_candidates.append((avg_len, col))
            else:
                if series.nunique() <= _CAT_MAX_CARDINALITY:
                    cat_cols.append(col)

    primary_text_col: Optional[str] = None
    if text_candidates:
        text_candidates.sort(reverse=True)
        primary_text_col = text_candidates[0][1]

    return num_cols, cat_cols, primary_text_col


def vectorize(df: pd.DataFrame, label_col: Optional[str] = None) -> FeatureBundle:
    """Превращает датафрейм в FeatureBundle с матрицей признаков X и вектором меток y."""
    if label_col is None:
        label_col = find_label_column(df)

    y: Optional[np.ndarray] = None
    if label_col and label_col in df.columns:
        y = df[label_col].values

    num_cols, cat_cols, text_col = _classify_columns(df, label_col)

    parts: List[np.ndarray] = []
    feature_names: List[str] = []
    meta: Dict[str, Any] = {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "text_column": text_col,
        "n_numeric": 0,
        "n_categorical": 0,
        "n_text_tfidf": 0,
    }

    if num_cols:
        X_num = df[num_cols].values.astype(float)
        X_num = SimpleImputer(strategy="median").fit_transform(X_num)
        X_num = StandardScaler().fit_transform(X_num)
        parts.append(X_num)
        feature_names.extend(num_cols)
        meta["n_numeric"] = len(num_cols)

    if cat_cols:
        # fillna перед astype(str) нужен для pandas 4: pd.NA → "<NA>", не "nan"
        X_cat_raw = (
            df[cat_cols]
            .fillna("__missing__")
            .astype(str)
            .values
        )
        X_cat_raw = np.where(
            np.isin(X_cat_raw, ["nan", "None", "NaN", "<NA>", ""]),
            "__missing__",
            X_cat_raw,
        )
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X_cat_raw).astype(float)
        # масштабируем, чтобы категориальные не доминировали в дистанционных методах
        X_cat = StandardScaler().fit_transform(X_cat)
        parts.append(X_cat)
        feature_names.extend(cat_cols)
        meta["n_categorical"] = len(cat_cols)

    if text_col:
        texts = df[text_col].fillna("").astype(str)
        tfidf = TfidfVectorizer(
            max_features=_TFIDF_MAX_FEATURES,
            sublinear_tf=True,   # log(1+tf) снижает вес частых слов
            ngram_range=(1, 1),  # только юниграммы — понятнее и быстрее
            min_df=2,            # убираем hapax legomena
            # стандартный token_pattern захватывает алфавитно-цифровые токены ≥ 2 символа,
            # включая смешанные вроде "qx9b7zftrigger"
        )
        X_text = tfidf.fit_transform(texts).toarray()
        parts.append(X_text)
        feature_names.extend([f"tfidf_{v}" for v in tfidf.get_feature_names_out()])
        meta["n_text_tfidf"] = int(X_text.shape[1])
        meta["tfidf_vocabulary_size"] = int(len(tfidf.vocabulary_))

    if not parts:
        raise ValueError(
            f"No usable feature columns found. "
            f"Numeric: {num_cols}, Categorical: {cat_cols}, Text: {text_col}."
        )

    X = np.hstack(parts).astype(np.float64)

    has_structured = (meta["n_numeric"] + meta["n_categorical"]) > 0
    has_text = meta["n_text_tfidf"] > 0
    if has_structured and has_text:
        dataset_type = "mixed"
    elif has_text:
        dataset_type = "text"
    else:
        dataset_type = "tabular"

    return FeatureBundle(
        X=X,
        y=y,
        label_col=label_col,
        feature_names=feature_names,
        dataset_type=dataset_type,
        meta=meta,
    )