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

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.generic_file_artifact_loader  import GenericFileArtifactLoader


class GenericFilePipelineModelLoader(GenericFileArtifactLoader):
    """
        Returns  Generic pipeline model instance associated with this model artifact.

        :return: pipeline model
        :rtype: spss.learn.Pipeline
        """
    def load_model(self):
        return(self.model_instance())


    def model_instance(self):
        """
         :return: returns Spss model path
         """
        return self.load()


    def pipeline_instance(self):
        """
         :return: returns Spss model path
         """
        return self.load()