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

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.hybrid_artifact_loader import HybridArtifactLoader


class HybridPipelineModelLoader(HybridArtifactLoader):
    """
        Returns pipeline model instance associated with this model artifact.

        :return: model
        :rtype: hybrid.model
        """
    def model_instance(self, artifact='full'):
        """
           :param artifact: query param string referring to "pipeline_model" or "full"
           Currently accepts:
           :return: returns a hybrid model content tar.gz file or pipeline_model.json
         """
        return self.load(artifact)
