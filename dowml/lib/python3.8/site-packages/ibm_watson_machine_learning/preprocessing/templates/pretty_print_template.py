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

template = """from ibm_watson_machine_learning.preprocessing import DataJoinGraph

data_join_graph = DataJoinGraph()

{nodes}

{edges}

data_join_graph.visualize()


from ibm_watson_machine_learning.experiment import AutoAI

experiment = AutoAI(wml_credentials, project_id=project_id)

pipeline_optimizer = experiment.optimizer(
    name={name},
    prediction_type={prediction_type},
    prediction_column={prediction_column},
    scoring={scoring},
    data_join_graph=data_join_graph,
    data_join_only=True
)


# Data join pipeline extracts features from paths:
{paths}
# Finally, selects deduplicated {nb_of_features} features.
# joined_data pandas dataframe can be used to work with joined data.

run_details = pipeline_optimizer.fit(
            training_data_reference=training_data_reference,
            background_mode=False)

joined_data = pipeline_optimizer.get_preprocessed_data_connection().read()
"""
