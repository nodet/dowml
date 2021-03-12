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

from abc import abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pandas import DataFrame
    from ibm_watson_machine_learning.utils.autoai.enums import PipelineTypes
    from sklearn.pipeline import Pipeline
    from numpy import ndarray

__all__ = [
    "BaseAutoPipelines"
]


class BaseAutoPipelines:
    """
    Base abstract class for Pipeline Optimizers.
    """

    @abstractmethod
    def get_params(self) -> dict:
        """Get configuration parameters of AutoPipelines"""
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> 'Pipeline':
        """Run fit job."""
        pass

    @abstractmethod
    def summary(self) -> 'DataFrame':
        """List all computed pipelines."""
        pass

    @abstractmethod
    def get_pipeline_details(self, pipeline_name: str = None) -> dict:
        """Get details of computed pipeline. Details like pipeline steps."""
        pass

    @abstractmethod
    def get_pipeline(self, pipeline_name: str, astype: 'PipelineTypes') -> Union['Pipeline', 'TrainablePipeline']:
        """Get particular computed Pipeline"""
        pass

    @abstractmethod
    def predict(self, X: Union['DataFrame', 'ndarray']) -> 'ndarray':
        """Use predict on top of the computed pipeline."""
        pass
