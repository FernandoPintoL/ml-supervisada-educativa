"""
Data Processor para Modelos Supervisados
Plataforma Educativa ML

Procesa datos crudos para entrenamiento:
- Limpieza de datos faltantes
- Normalización y escalado
- Manejo de valores atípicos
- Ingeniería de features
- Codificación de variables categóricas
"""

import logging
from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from shared.config import (
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    SUPERVISADO_FEATURES,
    DEBUG
)

# Configurar logger
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Procesador de datos para modelos supervisados.

    Proporciona:
    - Limpieza de datos
    - Normalización
    - Ingeniería de features
    - División train/val/test
    - Manejo de datos faltantes
    """

    def __init__(self, scaler_type: str = "standard", random_state: int = RANDOM_STATE):
        """
        Inicializar data processor.

        Args:
            scaler_type (str): 'standard' para StandardScaler, 'minmax' para MinMaxScaler
            random_state (int): Seed para reproducibilidad
        """
        self.scaler_type = scaler_type
        self.random_state = random_state

        # Inicializar escalador
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False

        logger.info(f"✓ DataProcessor inicializado (scaler: {scaler_type})")

    # ===========================================
    # LIMPIEZA DE DATOS
    # ===========================================

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remover filas duplicadas.

        Args:
            df (DataFrame): Datos
            subset (List[str]): Columnas para considerar duplicados (None = todas)

        Retorna:
            DataFrame: Datos sin duplicados
        """
        try:
            initial_rows = len(df)
            df_clean = df.drop_duplicates(subset=subset, keep='first')
            removed = initial_rows - len(df_clean)

            if removed > 0:
                logger.info(f"✓ Removidas {removed} filas duplicadas")

            return df_clean

        except Exception as e:
            logger.error(f"✗ Error removiendo duplicados: {str(e)}")
            return df

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = "mean",
                             threshold: float = 0.5) -> pd.DataFrame:
        """
        Manejar valores faltantes.

        Args:
            df (DataFrame): Datos con posibles NaN
            strategy (str): 'mean', 'median', 'forward_fill', 'drop'
            threshold (float): Si % NaN > threshold, dropar columna (0-1)

        Retorna:
            DataFrame: Datos sin NaN
        """
        try:
            # Calcular % de NaN por columna
            missing_percent = df.isnull().sum() / len(df)

            # Remover columnas con demasiados NaN
            cols_to_drop = missing_percent[missing_percent > threshold].index
            if len(cols_to_drop) > 0:
                logger.info(f"Removiendo {len(cols_to_drop)} columnas con > {threshold*100}% NaN")
                df = df.drop(columns=cols_to_drop)

            # Manejar valores faltantes restantes
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            if strategy == "drop":
                df = df.dropna()
            elif strategy == "forward_fill":
                df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
                df[categorical_cols] = df[categorical_cols].fillna(method='ffill')
            else:  # mean, median
                if numeric_cols.any():
                    imputer = SimpleImputer(strategy=strategy)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

                df[categorical_cols] = df[categorical_cols].fillna('unknown')

            logger.info(f"✓ Valores faltantes manejados (estrategia: {strategy})")
            return df

        except Exception as e:
            logger.error(f"✗ Error manejando valores faltantes: {str(e)}")
            return df

    def remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame:
        """
        Remover valores atípicos (outliers).

        Args:
            df (DataFrame): Datos
            columns (List[str]): Columnas a procesar (None = numéricas)
            method (str): 'iqr' o 'zscore'
            threshold (float): Para zscore (default 3.0 = 3 std)

        Retorna:
            DataFrame: Datos sin outliers
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            initial_rows = len(df)

            if method == "iqr":
                for col in columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            else:  # zscore
                for col in columns:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df = df[z_scores < threshold]

            removed = initial_rows - len(df)
            if removed > 0:
                logger.info(f"✓ Removidas {removed} filas con outliers (método: {method})")

            return df

        except Exception as e:
            logger.error(f"✗ Error removiendo outliers: {str(e)}")
            return df

    # ===========================================
    # NORMALIZACIÓN Y ESCALADO
    # ===========================================

    def fit_scalers(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> None:
        """
        Ajustar escaladores a los datos.

        Args:
            df (DataFrame): Datos
            numeric_cols (List[str]): Columnas numéricas a escalar
        """
        try:
            if numeric_cols is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                self.scaler.fit(df[numeric_cols])
                self.is_fitted = True
                logger.info(f"✓ Escaladores ajustados para {len(numeric_cols)} columnas")

        except Exception as e:
            logger.error(f"✗ Error ajustando escaladores: {str(e)}")

    def scale_data(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                  fit: bool = False) -> pd.DataFrame:
        """
        Escalar datos numéricos.

        Args:
            df (DataFrame): Datos
            numeric_cols (List[str]): Columnas a escalar
            fit (bool): Si True, ajusta el escalador también

        Retorna:
            DataFrame: Datos escalados
        """
        try:
            if numeric_cols is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            df_scaled = df.copy()

            if fit:
                self.fit_scalers(df, numeric_cols)

            if self.is_fitted and numeric_cols:
                df_scaled[numeric_cols] = self.scaler.transform(df[numeric_cols])
                logger.info(f"✓ Datos escalados ({self.scaler_type})")

            return df_scaled

        except Exception as e:
            logger.error(f"✗ Error escalando datos: {str(e)}")
            return df

    # ===========================================
    # CODIFICACIÓN DE VARIABLES
    # ===========================================

    def encode_categorical(self, df: pd.DataFrame,
                          categorical_cols: Optional[List[str]] = None,
                          fit: bool = False) -> pd.DataFrame:
        """
        Codificar variables categóricas (Label Encoding).

        Args:
            df (DataFrame): Datos
            categorical_cols (List[str]): Columnas categóricas
            fit (bool): Si True, ajusta los encoders también

        Retorna:
            DataFrame: Datos con variables codificadas
        """
        try:
            if categorical_cols is None:
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            df_encoded = df.copy()

            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()

                if fit:
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))

            if categorical_cols:
                logger.info(f"✓ Codificadas {len(categorical_cols)} variables categóricas")

            return df_encoded

        except Exception as e:
            logger.error(f"✗ Error codificando variables: {str(e)}")
            return df

    # ===========================================
    # INGENIERÍA DE FEATURES
    # ===========================================

    def select_features(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Seleccionar features específicas.

        Args:
            df (DataFrame): Datos completos
            features (List[str]): Features a seleccionar (None = usar SUPERVISADO_FEATURES)

        Retorna:
            DataFrame: Solo las features seleccionadas
        """
        try:
            if features is None:
                features = SUPERVISADO_FEATURES

            # Filtrar solo columnas que existen
            available_features = [f for f in features if f in df.columns]

            if len(available_features) < len(features):
                missing = set(features) - set(available_features)
                logger.warning(f"Features faltantes: {missing}")

            self.feature_names = available_features
            return df[available_features]

        except Exception as e:
            logger.error(f"✗ Error seleccionando features: {str(e)}")
            return df

    # ===========================================
    # PIPELINE COMPLETO
    # ===========================================

    def process(self, df: pd.DataFrame, target_col: Optional[str] = None,
               features: Optional[List[str]] = None,
               fit_scalers: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Procesar datos completos en un pipeline.

        Pasos:
        1. Remover duplicados
        2. Manejar valores faltantes
        3. Remover outliers
        4. Codificar variables categóricas
        5. Seleccionar features
        6. Escalar datos

        Args:
            df (DataFrame): Datos crudos
            target_col (str): Columna target (si existe)
            features (List[str]): Features a seleccionar
            fit_scalers (bool): Si True, ajusta scalers durante proceso

        Retorna:
            Tuple[DataFrame, Series]: (Features procesadas, Target si existe)
        """
        try:
            logger.info("Iniciando procesamiento de datos...")

            # Extraer target si existe
            target = None
            if target_col and target_col in df.columns:
                target = df[target_col].copy()
                df = df.drop(columns=[target_col])

            # 1. Limpieza
            df = self.remove_duplicates(df)
            df = self.handle_missing_values(df)
            df = self.remove_outliers(df)

            # 2. Codificación de categóricas
            df = self.encode_categorical(df, fit=fit_scalers)

            # 3. Seleccionar features
            df = self.select_features(df, features)

            # 4. Escalar
            df = self.scale_data(df, fit=fit_scalers)

            logger.info(f"✓ Datos procesados: {df.shape}")
            return df, target

        except Exception as e:
            logger.error(f"✗ Error en procesamiento completo: {str(e)}")
            return df, target

    # ===========================================
    # DIVISIÓN TRAIN/VAL/TEST
    # ===========================================

    def train_val_test_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                            test_size: float = TEST_SIZE,
                            val_size: float = VALIDATION_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                         pd.DataFrame,
                                                                         Optional[pd.Series],
                                                                         Optional[pd.Series],
                                                                         Optional[pd.Series]]:
        """
        Dividir datos en train, validation y test sets.

        Porcentajes:
        - Test: test_size (default 20%)
        - Validation: val_size de train (default 10% de 80% = 8%)
        - Train: resto (default 72%)

        Args:
            X (DataFrame): Features
            y (Series): Target
            test_size (float): Proporción de test (0-1)
            val_size (float): Proporción de validación del train (0-1)

        Retorna:
            Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            # Dividir train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            # Dividir train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state
            )

            logger.info(
                f"✓ Split realizado: "
                f"train={len(X_train)} ({len(X_train)/len(X)*100:.1f}%), "
                f"val={len(X_val)} ({len(X_val)/len(X)*100:.1f}%), "
                f"test={len(X_test)} ({len(X_test)/len(X)*100:.1f}%)"
            )

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"✗ Error dividiendo datos: {str(e)}")
            return X, pd.DataFrame(), pd.DataFrame(), y, None, None

    # ===========================================
    # UTILIDADES
    # ===========================================

    def get_feature_names(self) -> List[str]:
        """Obtener nombres de features procesadas."""
        return self.feature_names

    def __repr__(self) -> str:
        return f"DataProcessor(scaler={self.scaler_type}, features={len(self.feature_names)})"
