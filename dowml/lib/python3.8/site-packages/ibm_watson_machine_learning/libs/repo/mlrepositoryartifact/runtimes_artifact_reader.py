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

import logging

from  ibm_watson_machine_learning.libs.repo.mlrepository.artifact_reader import ArtifactReader

logger = logging.getLogger('RuntimesArtifactReader')


class RuntimesArtifactReader(ArtifactReader):
    def __init__(self, runtimespec_path):
        self.runtimespec_path = runtimespec_path

    def read(self):
        return self._open_stream()

    # This is a no. op. for RuntimeYmlFileReader as we do not want to delete the
    # archive file.
    def close(self):
        pass

    def _open_stream(self):
        return open(self.runtimespec_path, 'rt')

