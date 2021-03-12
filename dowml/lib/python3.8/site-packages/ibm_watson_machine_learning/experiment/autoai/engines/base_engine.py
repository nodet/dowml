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

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame
    from sklearn.pipeline import Pipeline
    from ibm_watson_machine_learning.helpers.connections import DataConnection

__all__ = [
    "BaseEngine"
]


class BaseEngine(ABC):
    """
    Base abstract class for Engines.
    """

    @abstractmethod
    def get_params(self) -> dict:
        """Fetch configuration parameters"""
        pass

    @abstractmethod
    def fit(self,
            training_data_reference: List['DataConnection'],
            training_results_reference: 'DataConnection',
            background_mode: bool = True) -> dict:
        """Schedule a fit/run/training."""
        pass

    @abstractmethod
    def get_run_status(self) -> str:
        """Fetch status of a training."""
        pass

    @abstractmethod
    def get_run_details(self) -> dict:
        """Fetch training details"""
        pass

    @abstractmethod
    def summary(self) -> 'DataFrame':
        """Fetch all pipelines results"""
        pass

    @abstractmethod
    def get_pipeline_details(self, pipeline_name: str = None) -> dict:
        """Fetch details of particular pipeline"""
        pass

    @abstractmethod
    def get_pipeline(self, pipeline_name: str, local_path: str = '.') -> 'Pipeline':
        """Download and load computed pipeline"""
        pass

    @abstractmethod
    def get_best_pipeline(self, local_path: str = '.') -> 'Pipeline':
        """Download and load the best pipeline"""
        pass
