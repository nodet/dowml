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

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.function_artifact_loader  import FunctionArtifactLoader


class FunctionLoader(FunctionArtifactLoader):
    """
        Returns  Generic function instance associated with this function artifact.

        :return: function
        :rtype:
        """

    def function_instance(self):
        """
         :return: returns function path
         """
        return self.load()

    def download_function(self,path):
        """
         :return: returns function path
         """
        return self.load(path)

