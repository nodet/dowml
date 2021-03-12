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

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps, WmlExperimentArtifact
from ibm_watson_machine_learning.libs.repo.util.exceptions import MetaPropMissingError


class ExperimentArtifact(WmlExperimentArtifact):
    """
    Class of Experiment artifacts created with MLRepositoryCLient.

    """
    def __init__(self,  uid=None, name=None, meta_props=MetaProps({})):
        super(ExperimentArtifact, self).__init__(uid, name, meta_props)


        if meta_props.prop(MetaNames.EXPERIMENTS.PATCH_INPUT) is None:
            if meta_props.prop(MetaNames.EXPERIMENTS.SETTINGS) is None :
                 raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
                                         '"MetaNames.EXPERIMENTS.SETTINGS"')

            if meta_props.prop(MetaNames.EXPERIMENTS.TRAINING_REFERENCES) is None :
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
                                       '"MetaNames.EXPERIMENTS.TRAINING_REFERENCES"')

            if meta_props.prop(MetaNames.EXPERIMENTS.TRAINING_DATA_REFERENCE) is None :
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
                                       '"MetaNames.EXPERIMENTS.TRAINING_DATA_REFERENCE"')


