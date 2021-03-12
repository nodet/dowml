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

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepository import ModelArtifact
from .python_version import PythonVersion
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate

import os, shutil, tarfile, json


class HybridPipelineModelArtifact(ModelArtifact):
    """
    Class of Hybrid model artifacts created with MLRepositoryCLient.

    """
    def __init__(self, hybrid_pipeline_model=None, uid=None, name=None, meta_props=MetaProps({})):
        super(HybridPipelineModelArtifact, self).__init__(uid, name, meta_props)

        self.ml_pipeline_model = hybrid_pipeline_model
        self.ml_pipeline = None     # no pipeline or parent reference

        if meta_props.prop(MetaNames.RUNTIMES) is None and meta_props.prop(MetaNames.RUNTIME_UID) is None and meta_props.prop(MetaNames.FRAMEWORK_RUNTIMES) is None:
            ver = PythonVersion.significant()
            runtimes = '[{"name":"python","version": "'+ ver + '"}]'
            self.meta.merge(
                MetaProps({MetaNames.FRAMEWORK_RUNTIMES: runtimes})
            )

    def pipeline_artifact(self):
        """
        Returns None. Pipeline is not implemented for Tensorflow model.
        """
        pass

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return HybridPipelineModelArtifact(
            self.ml_pipeline_model,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )


