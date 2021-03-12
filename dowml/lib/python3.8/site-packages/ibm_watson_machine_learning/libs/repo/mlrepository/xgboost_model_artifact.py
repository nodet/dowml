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

from ibm_watson_machine_learning.libs.repo.mlrepository import ModelArtifact


class XGBoostModelArtifact(ModelArtifact):
    """
    Class representing xgboost model artifact
    """
    def __init__(self, uid, name, meta_props):
        """
        Constructor for xgboost model artifact
        :param uid: unique id for xgboost model artifact
        :param name: name of the model
        :param metaprops: properties of the model and model artifact
        """
        super(XGBoostModelArtifact, self).__init__(uid, name, meta_props)