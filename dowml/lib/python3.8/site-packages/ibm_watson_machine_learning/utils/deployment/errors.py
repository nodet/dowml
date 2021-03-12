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

__all__ = [
    "WrongDeploymnetType",
    "ModelTypeNotSupported",
    "NotAutoAIExperiment",
    "EnvironmentNotSupported",
    'BatchJobFailed',
    'MissingScoringResults',
    'ModelStoringFailed',
    'DeploymentNotSupported',
    'MissingSpace'
]

from ibm_watson_machine_learning.utils import WMLClientError


class WrongDeploymnetType(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"This deployment is not of type: {value_name} ", reason)


class ModelTypeNotSupported(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"This model type is not supported yet: {value_name} ", reason)


class NotAutoAIExperiment(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"This experiment_run_id is not from an AutoAI experiment: {value_name} ", reason)


class EnvironmentNotSupported(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"This environment is not supported: {value_name}", reason)


class BatchJobFailed(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Batch job failed for job: {value_name}", reason)


class MissingScoringResults(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Scoring of deployment job: {value_name} not completed.", reason)


class ModelStoringFailed(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Model: {value_name} store failed.", reason)


class DeploymentNotSupported(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Deployment of type: {value_name} is not supported.", reason)


class MissingSpace(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Deployment needs to have space specified", reason)
