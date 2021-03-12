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


class ScikitModelArtifact(ModelArtifact):
    """
    Class representing model artifact.

    :param str uid: optional, uid which indicate that artifact already exists in repository service
    :param str name: optional, name of artifact
    :param MetaProps meta_props: optional, props used by other services
    """
    def __init__(self, uid, name, meta_props):
        super(ScikitModelArtifact, self).__init__(uid, name, meta_props)

    def pipeline_artifact(self):
        """
        Returns None. pipeline artifact for scikit model has not been implemented.

        :rtype: ModelArtifact
        """
        pass
