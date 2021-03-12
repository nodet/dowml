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
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pandas import DataFrame
    from ibm_watson_machine_learning.helpers import DataConnection

__all__ = [
    "BaseAutoPipelinesRuns"
]


class BaseAutoPipelinesRuns:
    """
    Base abstract class for Pipeline Optimizers Runs.
    """

    @abstractmethod
    def list(self) -> 'DataFrame':
        """Lists historical runs/fits with status."""
        pass

    @abstractmethod
    def get_params(self, run_id: str = None) -> dict:
        """Get executed optimizers configs parameters based on the run_id."""
        pass

    @abstractmethod
    def get_run_details(self, run_id: str = None) -> dict:
        """Get run details. If run_id is not supplied, last run will be taken."""
        pass

    @abstractmethod
    def get_optimizer(self, run_id: str):
        """Creates instance of AutoPipelinesRuns with all computed pipelines computed by AutoAi on WML."""
        pass

    @abstractmethod
    def get_data_connections(self, run_id: str) -> List['DataConnection']:
        """Create DataConnection objects for further user usage"""
        pass
