#  (C) Copyright IBM Corp. 2020.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from  ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.scikit_artifact_loader import ScikitArtifactLoader


class ScikitPipelineModelLoader(ScikitArtifactLoader):
    """
        Returns pipeline model instance associated with this model artifact.

        :return: pipeline model
        :rtype: scikit.learn.Pipeline
        """
    def model_instance(self,as_type=None):
        """
           :param as_type: string type referring to the model type to be returned.
           This parameter is applicable for xgboost models only.
           Currently accepts:
             'Booster': returns a model of type xgboost.Booster
             'XGBRegressor': returns a model of type xgboost.sklearn.XGBRegressor
           :return: returns a scikit model or an xgboost model of type xgboost.Booster
           or xgboost.sklearn.XGBRegressor
         """
        return self.load(as_type)
