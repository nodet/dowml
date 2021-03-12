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

import copy
from typing import List, Union

from ibm_watson_machine_learning.preprocessing import DataJoinGraph
from ibm_watson_machine_learning.utils.autoai.enums import (
    TShirtSize, ClassificationAlgorithms, RegressionAlgorithms, PredictionType, Metrics, Transformers,
    DataConnectionTypes, PipelineTypes, PositiveLabelClass)
from ibm_watson_machine_learning.utils.autoai.errors import LocalInstanceButRemoteParameter, MissingPositiveLabel
from ibm_watson_machine_learning.utils.autoai.utils import check_dependencies_versions,\
    validate_additional_params_for_optimizer
from ibm_watson_machine_learning.workspace import WorkSpace
from .engines import WMLEngine
from .optimizers import LocalAutoPipelines, RemoteAutoPipelines
from .runs import AutoPipelinesRuns, LocalAutoPipelinesRuns
from ..base_experiment.base_experiment import BaseExperiment

__all__ = [
    "AutoAI"
]


class AutoAI(BaseExperiment):
    """
    AutoAI class for pipeline models optimization automation.

    Parameters
    ----------
    wml_credentials: dictionary, required
        Credentials to Watson Machine Learning instance.

    project_id: str, optional
        ID of the Watson Studio project.

    space_id: str, optional
        ID of the Watson Studio Space.

    Example
    -------
    >>> from ibm_watson_machine_learning.experiment import AutoAI
    >>>
    >>> experiment = AutoAI(
    >>>        wml_credentials={
    >>>              "apikey": "...",
    >>>              "iam_apikey_description": "...",
    >>>              "iam_apikey_name": "...",
    >>>              "iam_role_crn": "...",
    >>>              "iam_serviceid_crn": "...",
    >>>              "instance_id": "...",
    >>>              "url": "https://us-south.ml.cloud.ibm.com"
    >>>            },
    >>>         project_id="...",
    >>>         space_id="...")
    """
    # note: initialization of AutoAI enums as class properties
    ClassificationAlgorithms = ClassificationAlgorithms
    RegressionAlgorithms = RegressionAlgorithms
    TShirtSize = TShirtSize
    PredictionType = PredictionType
    Metrics = Metrics
    Transformers = Transformers
    DataConnectionTypes = DataConnectionTypes
    PipelineTypes = PipelineTypes


    def __init__(self,
                 wml_credentials: Union[dict, 'WorkSpace'] = None,
                 project_id: str = None,
                 space_id: str = None) -> None:
        # note: as workspace is not clear enough to understand, there is a possibility to use pure
        # wml credentials with project and space IDs, but in addition we
        # leave a possibility to use a previous WorkSpace implementation, it could be passed as a first argument
        if wml_credentials is None:
            self._workspace = None
            self.runs = LocalAutoPipelinesRuns()

        else:
            if isinstance(wml_credentials, WorkSpace):
                self._workspace = wml_credentials
            else:
                self._workspace = WorkSpace(wml_credentials=wml_credentials.copy(),
                                            project_id=project_id,
                                            space_id=space_id)

            self.project_id = self._workspace.project_id
            self.space_id = self._workspace.space_id
            self.runs = AutoPipelinesRuns(engine=WMLEngine(self._workspace))
            self.runs._workspace = self._workspace

        self._20_class_limit_removal_test = False
        # --- end note

    def runs(self, *, filter: str) -> Union['AutoPipelinesRuns', 'LocalAutoPipelinesRuns']:
        """
        Get the historical runs but with WML Pipeline name filter (for remote scenario).
        Get the historical runs but with experiment name filter (for local scenario).

        Parameters
        ----------
        filter: str, required
            WML Pipeline name to filter the historical runs.
            or experiment name to filter the local historical runs.

        Returns
        -------
        AutoPipelinesRuns or LocalAutoPipelinesRuns

        Example
        -------
        >>> from ibm_watson_machine_learning.experiment import AutoAI
        >>> experiment = AutoAI(...)
        >>>
        >>> experiment.runs(filter='Test').list()
        """

        if self._workspace is None:
            return LocalAutoPipelinesRuns(filter=filter)

        else:
            return AutoPipelinesRuns(engine=WMLEngine(self._workspace.wml_client), filter=filter)

    def optimizer(self,
                  name: str,
                  *,
                  prediction_type: 'PredictionType',
                  prediction_column: str,
                  scoring: 'Metrics',
                  desc: str = None,
                  test_size: float = 0.1,
                  max_number_of_estimators: int = 2,
                  train_sample_rows_test_size: float = None,
                  daub_include_only_estimators: List[Union['ClassificationAlgorithms', 'RegressionAlgorithms']] = None,
                  cognito_transform_names: List['Transformers'] = None,
                  data_join_graph: 'DataJoinGraph' = None,
                  csv_separator: Union[List[str], str] = ',',
                  excel_sheet: Union[str, int] = 0,
                  encoding: str = 'utf-8',
                  positive_label: str = None,
                  data_join_only: bool = False,
                  drop_duplicates: bool = True,
                  **kwargs) -> Union['RemoteAutoPipelines', 'LocalAutoPipelines']:
        """
        Initialize an AutoAi optimizer.

        Parameters
        ----------
        name: str, required
            Name for the AutoPipelines

        prediction_type: PredictionType, required
            Type of the prediction.

        prediction_column: str, required
            name of the target/label column

        scoring: Metrics, required
            Type of the metric to optimize with.

        desc: str, optional
            Description

        test_size: float, optional
            Percentage of the entire dataset to leave as a holdout. Default 0.1

        max_number_of_estimators: int, optional
            Maximum number (top-K ranked by DAUB model selection) of the selected algorithm, or estimator types,
            for example LGBMClassifierEstimator, XGBoostClassifierEstimator, or LogisticRegressionEstimator
            to use in pipeline composition.  The default is 2, where only the highest ranked by model
            selection algorithm type is used. (min 1, max 4)

        train_sample_rows_test_size: float, optional
            Training data sampling percentage

        daub_include_only_estimators: List[Union['ClassificationAlgorithms', 'RegressionAlgorithms']], optional
            List of estimators to include in computation process.
            See: AutoAI.ClassificationAlgorithms or AutoAI.RegressionAlgorithms

        cognito_transform_names: List['Transformers'], optional
            List of transformers to include in feature enginnering computation process.
            See: AutoAI.Transformers

        csv_separator: Union[List[str], str], optional
            The separator, or list of separators to try for separating
            columns in a CSV file.  Not used if the file_name is not a CSV file.
            Default is ','.

        excel_sheet: Union[str, int], optional
            Name or number of the excel sheet to use. Only use when xlsx file is an input.
            Default is 0.

        encoding: str, optional
            Encoding type for CSV training file.

        positive_label: str, optional
            The positive class to report when binary classification.
            When multiclass or regression, this will be ignored.

        t_shirt_size: TShirtSize, optional
            The size of the remote AutoAI POD instance (computing resources). Only applicable to a remote scenario.
            See: AutoAI.TShirtSize

        data_join_graph: DataJoinGraph, optional
            A graph object with definition of join structure for multiple input data sources.
            Data preprocess step for multiple files.

        data_join_only: bool, optional
            If True only preprocessing will be executed.

        drop_duplicates: bool, optional
            If True duplicated rows in data will be removed before further processing. Default is True.

        Returns
        -------
        RemoteAutoPipelines or LocalAutoPipelines, depends on how you initialize the AutoAI object.

        Example
        -------
        >>> from ibm_watson_machine_learning.experiment import AutoAI
        >>> experiment = AutoAI(...)
        >>>
        >>> optimizer = experiment.optimizer(
        >>>        name="name of the optimizer.",
        >>>        prediction_type=AutoAI.PredictionType.BINARY,
        >>>        prediction_column="y",
        >>>        scoring=AutoAI.Metrics.ROC_AUC_SCORE,
        >>>        desc="Some description.",
        >>>        test_size=0.1,
        >>>        max_num_daub_ensembles=1,
        >>>        cognito_transform_names=[AutoAI.Transformers.SUM,AutoAI.Transformers.MAX],
        >>>        train_sample_rows_test_size=1,
        >>>        daub_include_only_estimators=[AutoAI.ClassificationAlgorithms.LGBM, AutoAI.ClassificationAlgorithms.XGB],
        >>>        t_shirt_size=AutoAI.TShirtSize.L
        >>>    )
        >>>
        >>> optimizer = experiment.optimizer(
        >>>        name="name of the optimizer.",
        >>>        prediction_type=AutoAI.PredictionType.MULTICLASS,
        >>>        prediction_column="y",
        >>>        scoring=AutoAI.Metrics.ROC_AUC_SCORE,
        >>>        desc="Some description.",
        >>>    )
        """
        if data_join_graph:
            data_join_graph.problem_type = prediction_type
            data_join_graph.target_column = prediction_column

        if (prediction_type == PredictionType.BINARY and scoring in vars(PositiveLabelClass).values()
                and positive_label is None):
            raise MissingPositiveLabel(scoring, reason=f"\"{scoring}\" needs a \"positive_label\" "
                                                       f"parameter to be defined when used with binary classification.")

        if self._workspace is None and kwargs.get('t_shirt_size'):
            raise LocalInstanceButRemoteParameter(
                "t_shirt_size",
                reason="During LocalOptimizer initialization, \"t_shirt_size\" parameter was provided. "
                       "\"t_shirt_size\" parameter is only applicable to the RemoteOptimizer instance."
            )
        elif self._workspace is None:
            reduced_kwargs = copy.copy(kwargs)

            for n in ['_force_local_scenario']:
                if n in reduced_kwargs:
                    del reduced_kwargs[n]

            validate_additional_params_for_optimizer(reduced_kwargs)

            return LocalAutoPipelines(
                name=name,
                prediction_type='classification' if prediction_type in ['binary', 'multiclass'] else prediction_type,
                prediction_column=prediction_column,
                scoring=scoring,
                desc=desc,
                test_size=test_size,
                max_num_daub_ensembles=max_number_of_estimators,
                train_sample_rows_test_size=train_sample_rows_test_size,
                daub_include_only_estimators=daub_include_only_estimators,
                cognito_transform_names=cognito_transform_names,
                positive_label=positive_label,
                _force_local_scenario=kwargs.get('_force_local_scenario', False),
                **reduced_kwargs
            )

        else:
            reduced_kwargs = copy.copy(kwargs)

            for n in ['t_shirt_size', 'notebooks', 'autoai_pod_version', 'obm_pod_version']:
                if n in reduced_kwargs:
                    del reduced_kwargs[n]

            validate_additional_params_for_optimizer(reduced_kwargs)

            engine = WMLEngine(self._workspace)

            if self._20_class_limit_removal_test:
                engine._20_class_limit_removal_test = True

            optimizer = RemoteAutoPipelines(
                name=name,
                prediction_type=prediction_type,
                prediction_column=prediction_column,
                scoring=scoring,
                desc=desc,
                test_size=test_size,
                max_num_daub_ensembles=max_number_of_estimators,
                t_shirt_size=self._workspace.restrict_pod_size(t_shirt_size=kwargs.get(
                    't_shirt_size', TShirtSize.M if self._workspace.wml_client.ICP else TShirtSize.L)
                ),
                train_sample_rows_test_size=train_sample_rows_test_size,
                daub_include_only_estimators=daub_include_only_estimators,
                cognito_transform_names=cognito_transform_names,
                data_join_graph=data_join_graph,
                drop_duplicates=drop_duplicates,
                csv_separator=csv_separator,
                excel_sheet=excel_sheet,
                encoding=encoding,
                positive_label=positive_label,
                data_join_only=data_join_only,
                engine=engine,
                notebooks=kwargs.get('notebooks', True),
                autoai_pod_version=kwargs.get('autoai_pod_version', None),
                obm_pod_version=kwargs.get('obm_pod_version', None),
                **reduced_kwargs
            )
            optimizer._workspace = self._workspace
            return optimizer
