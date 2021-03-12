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

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.tensorflow_artifact_loader import TensorflowArtifactLoader
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[TENSORFLOW]:
    import tensorflow as tf

class TensorflowPipelineModelLoader(TensorflowArtifactLoader):
    """
        Returns Tensorflow Runtime  instance associated with this model artifact.

        :return: TensorflowArtifactRunTime instance
        :rtype: TensorflowRuntimeArtifact
        """
    def load_model(self,session=None,tags=None):
        return(self.model_instance(session,tags))



    def model_instance(self,session=None,tags=None):
        lib_checker.check_lib(TENSORFLOW)
        if session is None:
            if '2.1.0' in tf.__version__:
                session = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
            else:
                session = tf.Session(graph=tf.Graph())
        elif session is not None:
            if '2.1.0' in tf.__version__ and not isinstance(session, tf.compat.v1.Session):
                raise TypeError("sess should be of type : %s" % tf.compat.v1.Session)
            elif not isinstance(session,tf.Session):
                raise TypeError("sess should be of type : %s" % tf.Session)

        if tags is None:
            if '2.1.0' in tf.__version__:
                tags = [tf.compat.v1.saved_model.tag_constants.SERVING]
            else:
                tags = [tf.saved_model.tag_constants.SERVING]
        elif tags is not None:
            if not isinstance(tags,list):
                raise TypeError("tags should be of type : %s" % list)


        return self.load(session,tags)
