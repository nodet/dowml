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

import mlpipelinepy


class MLPipelineVersion(type):
    _version = mlpipelinepy.__version__
    _version_parts = _version.split(".")

    @classmethod
    def major_ver(mcs):
        return int(mcs._version_parts[0])

    @classmethod
    def minor_ver(mcs):
        return int(mcs._version_parts[1])

    @classmethod
    def full_version(mcs):
        return mcs._version
