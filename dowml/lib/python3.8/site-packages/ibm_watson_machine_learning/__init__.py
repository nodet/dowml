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

"""Package skeleton

.. moduleauthor:: Wojciech Sobala <wojciech.sobala@pl.ibm.com>
"""

from os.path import join as path_join
import pkg_resources
import sys

try:
    wml_location = pkg_resources.get_distribution("ibm-watson-machine-learning").location
    sys.path.insert(1, path_join(wml_location, 'ibm_watson_machine_learning', 'libs'))
    sys.path.insert(2, path_join(wml_location, 'ibm_watson_machine_learning', 'tools'))
except pkg_resources.DistributionNotFound:
    pass
from ibm_watson_machine_learning.utils import version
from ibm_watson_machine_learning.client import APIClient

from .utils import is_python_2
if is_python_2():
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Python 2 is not officially supported.")
