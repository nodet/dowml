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

__all__ = [
    'is_run_id_exists'
]

from typing import Dict, Optional

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure


def is_run_id_exists(wml_credentials: Dict, run_id: str, space_id: Optional[str] = None) -> bool:
    """
    Check if specified run_id exists for WML client initialized with passed credentials.

    Parameters
    ----------
    wml_credentials: dictionary, required

    run_id: str, required
        Training run id of AutoAI experiment.

    space_id: str, optional
        Optional space id for WMLS and CP4D.
    """
    client = APIClient(wml_credentials)

    if space_id is not None:
        client.set.default_space(space_id)

    try:
        client.training.get_details(run_id)

    except ApiRequestFailure as e:
        if 'Status code: 404' in str(e):
            return False

        else:
            raise e

    return True
