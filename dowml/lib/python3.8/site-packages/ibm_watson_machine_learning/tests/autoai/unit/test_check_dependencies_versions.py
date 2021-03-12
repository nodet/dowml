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

import unittest

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.utils.autoai.utils import check_dependencies_versions
from ibm_watson_machine_learning.tests.utils import get_wml_credentials


class MyTestCase(unittest.TestCase):
    request_json = {'hybrid_pipeline_software_specs': [{'name': "autoai-kb_3.1-py3.7"}]}

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

    def test_01__all_and_xgboost(self):
        check_dependencies_versions(self.request_json, self.wml_client, estimator_pkg='xgboost')

    def test_02__all_and_lightgbm(self):
        check_dependencies_versions(self.request_json, self.wml_client, estimator_pkg='lightgbm')


if __name__ == '__main__':
    unittest.main()
