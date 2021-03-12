# (C) Copyright IBM Corp. 2020.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

__all__ = [
    "ClassificationAlgorithms",
    "RegressionAlgorithms",
    "PredictionType",
    "Metrics",
    "Transformers",
    "DataConnectionTypes",
    "RunStateTypes",
    "PipelineTypes",
    "Directions",
    "TShirtSize",
    "MetricsToDirections",
    'PositiveLabelClass',
    'VisualizationTypes'
]


class ClassificationAlgorithms(Enum):
    """Classification algorithms that AutoAI could use."""
    EX_TREES = "ExtraTreesClassifierEstimator"
    GB = "GradientBoostingClassifierEstimator"
    LGBM = "LGBMClassifierEstimator"
    LR = "LogisticRegressionEstimator"
    RF = "RandomForestClassifierEstimator"
    XGB = "XGBClassifierEstimator"
    DT = "DecisionTreeClassifierEstimator"


class RegressionAlgorithms(Enum):
    """Regression algorithms that AutoAI could use."""
    RF = "RandomForestRegressorEstimator"
    RIDGE = "RidgeEstimator"
    EX_TREES = "ExtraTreesRegressorEstimator"
    GB = "GradientBoostingRegressorEstimator"
    LR = "LinearRegressionEstimator"
    XGB = "XGBRegressorEstimator"
    LGBM = "LGBMRegressorEstimator"
    DT = "DecisionTreeRegressorEstimator"


class PredictionType:
    """
        Supported types of learning.
        OneOf: [BINARY, MULTICLASS, REGRESSION]
    """
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class PositiveLabelClass:
    """Metrics that need positive label definition for binary classification."""
    AVERAGE_PRECISION_SCORE = "average_precision"
    F1_SCORE = "f1"
    PRECISION_SCORE = "precision"
    RECALL_SCORE = "recall"
    F1_SCORE_MICRO = "f1_micro"
    F1_SCORE_MACRO = "f1_macro"
    F1_SCORE_WEIGHTED = "f1_weighted"
    PRECISION_SCORE_MICRO = "precision_micro"
    PRECISION_SCORE_MACRO = "precision_macro"
    PRECISION_SCORE_WEIGHTED = "precision_weighted"
    RECALL_SCORE_MICRO = "recall_micro"
    RECALL_SCORE_MACRO = "recall_macro"
    RECALL_SCORE_WEIGHTED = "recall_weighted"


class Metrics:
    """
        Supported types of classification and regression metrics in autoai.

    """
    ACCURACY_SCORE = "accuracy"
    AVERAGE_PRECISION_SCORE = "average_precision"
    F1_SCORE = "f1"
    LOG_LOSS = "neg_log_loss"
    PRECISION_SCORE = "precision"
    RECALL_SCORE = "recall"
    ROC_AUC_SCORE = "roc_auc"

    F1_SCORE_MICRO = "f1_micro"
    F1_SCORE_MACRO = "f1_macro"
    F1_SCORE_WEIGHTED = "f1_weighted"
    PRECISION_SCORE_MICRO = "precision_micro"
    PRECISION_SCORE_MACRO = "precision_macro"
    PRECISION_SCORE_WEIGHTED = "precision_weighted"
    RECALL_SCORE_MICRO = "recall_micro"
    RECALL_SCORE_MACRO = "recall_macro"
    RECALL_SCORE_WEIGHTED = "recall_weighted"

    EXPLAINED_VARIANCE_SCORE = "explained_variance"
    MEAN_ABSOLUTE_ERROR = "neg_mean_absolute_error"
    MEAN_SQUARED_ERROR = "neg_mean_squared_error"
    MEAN_SQUARED_LOG_ERROR = "neg_mean_squared_log_error"
    MEDIAN_ABSOLUTE_ERROR = "neg_median_absolute_error"
    ROOT_MEAN_SQUARED_ERROR = "neg_root_mean_squared_error"
    ROOT_MEAN_SQUARED_LOG_ERROR = "neg_root_mean_squared_log_error"
    R2_SCORE = "r2"


class Transformers:
    """
        Supported types of congito transformers names in autoai.
    """
    SQRT = "sqrt"
    LOG = "log"
    ROUND = "round"
    SQUARE = "square"
    CBRT = "cbrt"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"

    ABS = "abs"
    SIGMOID = "sigmoid"
    PRODUCT = "product"
    MAX = "max"
    DIFF = "diff"
    SUM = "sum"
    DIVIDE = "divide"
    STDSCALER = "stdscaler"

    MINMAXSCALER = "minmaxscaler"
    PCA = "pca"
    NXOR = "nxor"
    CUBE = "cube"
    FEATUREAGGLOMERATION = "featureagglomeration"
    ISOFORESTANOMALY = "isoforestanomaly"


class DataConnectionTypes:
    """
        Supported types of DataConnection.
        OneOf: [s3, FS]
    """
    S3 = "s3"
    FS = 'fs'
    DS = 'data_asset'
    CA = 'connection_asset'


class RunStateTypes:
    """
        Supported types of AutoAI fit/run.
    """
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineTypes:
    """
        Supported types of Pipelines.
    """
    LALE = "lale"
    SKLEARN = "sklearn"


class Directions:
    """Possible metrics directions"""
    ASCENDING = "ascending"
    DESCENDING = "descending"


class TShirtSize:
    """
    Possible sizes of the AutoAI POD
    Depends on the POD size, AutoAI could support different data sets sizes.

    S - small (2vCPUs and 8GB of RAM)
    M - Medium (4vCPUs and 16GB of RAM)
    L - Large (8vCPUs and 32GB of RAM))
    XL - Extra Large (16vCPUs and 64GB of RAM)
    """
    S = 's'
    M = 'm'
    ML = 'ml'
    L = 'l'
    XL = 'xl'


class MetricsToDirections(Enum):
    """Map of metrics directions."""
    ROC_AUC = Directions.ASCENDING
    NORMALIZED_GINI_COEFFICIENT = Directions.ASCENDING
    PRECISION = Directions.ASCENDING
    AVERAGE_PRECISION = Directions.ASCENDING
    NEG_LOG_LOSS = Directions.DESCENDING
    RECALL = Directions.ASCENDING
    ACCURACY = Directions.ASCENDING
    F1 = Directions.ASCENDING

    PRECISION_MICRO = Directions.ASCENDING
    PRECISION_MACRO = Directions.ASCENDING
    PRECISION_WEIGHTED = Directions.ASCENDING
    F1_MICRO = Directions.ASCENDING
    F1_MACRO = Directions.ASCENDING
    F1_WEIGHTED = Directions.ASCENDING
    RECALL_MICRO = Directions.ASCENDING
    RECALL_MACRO = Directions.ASCENDING
    RECALL_WEIGHTED = Directions.ASCENDING

    NEG_ROOT_MEAN_SQUARED_ERROR = Directions.DESCENDING
    EXPLAINED_VARIANCE = Directions.ASCENDING
    NEG_MEAN_ABSOLUTE_ERROR = Directions.DESCENDING
    NEG_MEAN_SQUARED_ERROR = Directions.DESCENDING
    NEG_MEAN_SQUARED_LOG_ERROR = Directions.DESCENDING
    NEG_MEDIAN_ABSOLUTE_ERROR = Directions.DESCENDING
    NEG_ROOT_MEAN_SQUARED_LOG_ERROR = Directions.DESCENDING
    R2 = Directions.ASCENDING


class VisualizationTypes:
    """Types of visualization options."""
    PDF = 'pdf'
    INPLACE = 'inplace'
