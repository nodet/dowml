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

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepository.wml_function_artifact import WmlFunctionArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.function_artifact_reader import FunctionArtifactReader

class FunctionArtifact(WmlFunctionArtifact):
    """
    Class of  function artifacts created with MLRepositoryCLient.

    """
    def __init__(self,
                 function,
                 uid=None,
                 name=None,
                 meta_props=MetaProps({}),):

        super(FunctionArtifact, self).__init__(uid, name, meta_props)

        self.function = function

    def reader(self):
        """
        Returns reader used for getting archive model content.

        :return: reader for TensorflowPipelineModelArtifact.pipeline.Pipeline
        :rtype: TensorflowPipelineReader
        """
        try:
            return self._reader
        except:
            self._reader = FunctionArtifactReader(self.function)
            return self._reader

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return FunctionArtifact(
            self.function,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )

