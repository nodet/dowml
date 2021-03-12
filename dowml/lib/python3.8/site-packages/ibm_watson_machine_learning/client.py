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

import logging
from ibm_watson_machine_learning.utils import version

#from ibm_watson_machine_learning.learning_system import LearningSystem
from ibm_watson_machine_learning.experiments import Experiments
from ibm_watson_machine_learning.repository import Repository
from ibm_watson_machine_learning.model_definition import ModelDefinition
from ibm_watson_machine_learning.models import Models
from ibm_watson_machine_learning.pipelines import Pipelines
from ibm_watson_machine_learning.instance import ServiceInstance
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.training import Training
from ibm_watson_machine_learning.runtimes import Runtimes
from ibm_watson_machine_learning.functions import Functions
from ibm_watson_machine_learning.spaces import Spaces
from ibm_watson_machine_learning.platform_spaces import PlatformSpaces
from ibm_watson_machine_learning.assets import Assets
from ibm_watson_machine_learning.connections import Connections
from ibm_watson_machine_learning.Set import Set
from ibm_watson_machine_learning.sw_spec import SwSpec
from ibm_watson_machine_learning.hw_spec import HwSpec
from ibm_watson_machine_learning.pkg_extn import PkgExtn
from ibm_watson_machine_learning.shiny import Shiny
from ibm_watson_machine_learning.script import Script
from ibm_watson_machine_learning.wml_client_error import NoWMLCredentialsProvided
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.instance_new_plan import ServiceInstanceNewPlan

from ibm_watson_machine_learning.migration_v4ga_cloud import Migrationv4GACloud
from ibm_watson_machine_learning.remote_training_system import RemoteTrainingSystem

from ibm_watson_machine_learning.export_assets import Export
from ibm_watson_machine_learning.import_assets import Import
from ibm_watson_machine_learning.volumes import Volume

import copy
import os
import requests
import sys


'''
.. module:: APIClient
   :platform: Unix, Windows
   :synopsis: Watson Machine Learning API Client.

.. moduleauthor:: IBM
'''


class APIClient:

    def __init__(self, wml_credentials, project_id=None):
        self._logger = logging.getLogger(__name__)
        self.wml_credentials = copy.deepcopy(wml_credentials)
        self.CAMS = None
        self.WSD = None
        self.WSD_20 = None
        self.ICP_30 = None
        self.ICP = None
        self.default_space_id = None
        self.default_project_id = None
        self.CLOUD_PLATFORM_SPACES = False
        self.PLATFORM_URL = None
        self.CAMS_URL = None
        self.CREATED_IN_V1_PLAN = False
        self.version_param = '2020-08-01'
        self.ICP_PLATFORM_SPACES = False # This will be applicable for 3.5 and later and specific to convergence functionalities
        self.ICP_35 = False # Use it for any 3.5 specific functionalities
        self.CLOUD_BETA_FLOW = False
        # self.version_param = '2020-06-12'

        self.PLATFORM_URLS_MAP = {
            'https://wml-dev.ml.test.cloud.ibm.com': 'https://api.dataplatform.dev.cloud.ibm.com',
            'https://wml-fvt.ml.test.cloud.ibm.com': 'https://api.dataplatform.dev.cloud.ibm.com',
            'https://us-south.ml.test.cloud.ibm.com': 'https://api.dataplatform.dev.cloud.ibm.com',
            'https://yp-qa.ml.cloud.ibm.com': 'https://api.dataplatform.test.cloud.ibm.com',
            'https://private.yp-qa.ml.cloud.ibm.com': 'https://api.dataplatform.test.cloud.ibm.com',
            'https://yp-cr.ml.cloud.ibm.com': 'https://api.dataplatform.test.cloud.ibm.com',
            'https://private.yp-cr.ml.cloud.ibm.com': 'https://api.dataplatform.test.cloud.ibm.com',
            'https://jp-tok.ml.cloud.ibm.com': 'https://api.jp-tok.dataplatform.cloud.ibm.com',
            'https://eu-gb.ml.cloud.ibm.com': 'https://api.eu-gb.dataplatform.cloud.ibm.com',
            'https://eu-de.ml.cloud.ibm.com': 'https://api.eu-de.dataplatform.cloud.ibm.com',
            'https://us-south.ml.cloud.ibm.com': 'https://api.dataplatform.cloud.ibm.com'
        }

        if "token" in wml_credentials:
            self.proceed = True
        else:
            self.proceed = False

        self.project_id = project_id
        self.wml_token = None
        self.__wml_local_supported_version_list = ['1.0', '1.1',
                                                   '2.0', '2.0.1', '2.0.2', '2.5.0',
                                                   '3.0.0', '3.0.1', '3.5']
        self.__wsd_supported_version_list = ['1.0', '1.1', '2.0']
        self.__predefined_instance_type_list = ['icp', 'openshift', 'wml_local', 'wsd_local']
        os.environ['WSD_PLATFORM'] = 'False'
        if wml_credentials is None:
            raise NoWMLCredentialsProvided()
        if self.wml_credentials['url'][-1] == "/":
            self.wml_credentials['url'] = self.wml_credentials['url'].rstrip('/')

        if 'instance_id' not in wml_credentials.keys():
            self.CLOUD_PLATFORM_SPACES = True
            self.wml_credentials[u'instance_id'] = 'invalid'  # This is applicable only via space instance_id
            self.ICP = False
            self.CAMS = False

            if wml_credentials[u'url'] in self.PLATFORM_URLS_MAP.keys():
                self.PLATFORM_URL = self.PLATFORM_URLS_MAP[wml_credentials[u'url']]
                self.CAMS_URL = self.PLATFORM_URLS_MAP[wml_credentials[u'url']]
            else:
                raise WMLClientError("Provided 'url' is not valid")

            if not self._is_IAM():
                raise WMLClientError('apikey for IAM token is not provided in credentials for the client')
        else:
            if 'icp' == wml_credentials[u'instance_id'].lower() or 'openshift' == wml_credentials[
                u'instance_id'].lower() or 'wml_local' == wml_credentials[u'instance_id'].lower():
                self.ICP = True
                os.environ["DEPLOYMENT_PLATFORM"] = "private"
                ##Condition for CAMS related changes to take effect (Might change)
                if 'version' in wml_credentials.keys() and (
                        '2.0.1' == wml_credentials[u'version'].lower() or '2.5.0' == wml_credentials[
                    u'version'].lower() or
                        '3.0.0' == wml_credentials[u'version'].lower() or '3.0.1' == wml_credentials[
                            u'version'].lower() or '3.5' == wml_credentials[u'version'].lower() or
                        '1.1' == wml_credentials[u'version'].lower() or '2.0' == wml_credentials[u'version'].lower()):
                    self.CAMS = True
                    os.environ["DEPLOYMENT_PRIVATE"] = "icp4d"
                    if 'wml_local' == wml_credentials[u'instance_id'].lower() and \
                            ('1.1' == wml_credentials[u'version'].lower() or '2.0' == wml_credentials[
                                u'version'].lower()):
                        url_port = wml_credentials[u'url'].split(':')[-1]
                        if not url_port.isdigit():
                            raise WMLClientError("It is mandatory to have port number as part of url for wml_local.")

                    if '3.0.0' == wml_credentials[u'version'].lower() or \
                            '3.0.1' == wml_credentials[u'version'].lower() or \
                            '2.0' == wml_credentials[u'version'].lower():
                        self.ICP_30 = True

                    # For Cloud convergence related functionalities brought into CP4D 3.5
                    if '3.5' == wml_credentials[u'version'].lower():
                        self.ICP_PLATFORM_SPACES = True
                        self.ICP_35 = True

                else:
                    if 'version' in wml_credentials.keys() and \
                            wml_credentials[u'version'].lower() not in self.__wml_local_supported_version_list:
                        raise WMLClientError(
                            "Invalid value for 'version' provided in wml_credentials. Please check the wml_credentials provided." +
                            "Supported value for version field are: " + ', '.join(
                                self.__wml_local_supported_version_list))

                    self.CAMS = False
            else:
                if ('wsd_local' == wml_credentials[u'instance_id'].lower()) and \
                        ('1.1' == wml_credentials[u'version'].lower() or '2.0' == wml_credentials[u'version'].lower()):
                    self.WSD = True
                    os.environ['WSD_PLATFORM'] = 'True'
                    if '2.0' == wml_credentials[u'version'].lower():
                        self.WSD_20 = True
                else:
                    if ('wsd_local' == wml_credentials[u'instance_id'].lower()) and \
                            'version' in wml_credentials.keys() and \
                            wml_credentials[u'version'].lower() not in self.__wsd_supported_version_list:
                        raise WMLClientError(
                            "Invalid value for 'version' provided in wml_credentials. Please check the wml_credentials provided." +
                            "Supported value for version field are: " + ', '.join(self.__wsd_supported_version_list))

                    self.ICP = False
                    self.CAMS = False

                    self.service_instance = ServiceInstanceNewPlan(self)

                    headers = self._get_headers()

                    del headers[u'X-WML-User-Client']
                    if 'ML-Instance-ID' in headers:
                        del headers[u'ML-Instance-ID']
                    del headers[u'x-wml-internal-switch-to-new-v4']

                    response_get_instance = requests.get(
                        u'{}/ml/v4/instances/{}'.format(self.wml_credentials['url'],
                                                  self.wml_credentials['instance_id']),
                        params={'version': self.version_param},
                        # params=self._client._params(),
                        headers=headers
                    )

                    if response_get_instance.status_code == 200:
                        if 'plan' in response_get_instance.json()[u'entity']:
                            tags = response_get_instance.json()[u'metadata']['tags']
                            if 'created_in_v1_plan' in tags:
                                self.CREATED_IN_V1_PLAN = True
                            if response_get_instance.json()[u'entity'][u'plan'][u'version'] == 2 and\
                                    not self.CREATED_IN_V1_PLAN and\
                                    response_get_instance.json()[u'entity'][u'plan']['name'] != 'lite':
                                self.CLOUD_PLATFORM_SPACES = True

                                # If v1 lite converted to v2 plan, tags : ["created_in_v1_plan"]

                                if not self._is_IAM():
                                    raise WMLClientError('apikey for IAM token is not provided in credentials for '
                                                         'the client.')

                                if wml_credentials[u'url'] in self.PLATFORM_URLS_MAP.keys():
                                    self.PLATFORM_URL = self.PLATFORM_URLS_MAP[wml_credentials[u'url']]
                                    self.CAMS_URL = self.PLATFORM_URLS_MAP[wml_credentials[u'url']]
                                else:
                                    raise WMLClientError("No 'url' provided")

                                if self.CLOUD_PLATFORM_SPACES:
                                    print("Note that if you are using any of v2 instance plans, then 'instance_id' is"
                                          " not required to be provided here. It will be picked up from space when "
                                          "client.set.default_space(space_id) is called ")

                            if not self.CLOUD_PLATFORM_SPACES:
                                self.CLOUD_BETA_FLOW = True
                                print("NOTE!! DEPRECATED!! Creating assets using this flow(v4 beta) is deprecated"
                                      " starting Sep 1st, 2020 and will be discontinued at the end of the migration period. "
                                      "Refer to the documentation at 'https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/wml-ai.html' "
                                      "for the migration process to be able to access new features")
                    elif response_get_instance.status_code == 404:
                        raise WMLClientError("instance_id not found. Note that if you are using "
                                             "any of new instance plans [], then 'instance_id' is not required "
                                             "to be provided here. It will be picked up from space when "
                                             "client.set.default_space(space_id) is called ")
                    elif response_get_instance.status_code == 401:
                        raise WMLClientError("Not authorized to access the instance_id  Note that if you are using "
                                             "any of new instance plans [], then 'instance_id' is not required "
                                             "to be provided here. It will be picked up from space when "
                                             "client.set.default_space(space_id) is called")
                    else:
                        # if someone is using beta flow and using instance api key
                        response_get_instance_v1 = requests.get(
                            u'{}/v3/wml_instances/{}'.format(self.wml_credentials['url'],
                                                             self.wml_credentials['instance_id']),
                            headers=self._get_headers()
                        )

                        if response_get_instance_v1.status_code != 200:
                            raise WMLClientError("Failed to get instance details. Note that if you are using "
                                                 "any of new instance plans [], then 'instance_id' is not required "
                                                 "to be provided here. It will be picked up from space when "
                                                 "client.set.default_space(space_id) is called.\n  Error:" + response_get_instance.text)

                        self.CLOUD_BETA_FLOW = True
                        print("NOTE!! DEPRECATED!! Creating assets using this flow(v4 beta) is deprecated"
                              " starting Sep 1st, 2020 and will be discontinued at the end of the migration period. "
                              "Refer to the documentation at 'https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/wml-ai.html' "
                              "for the migration process to be able to access new features")
            if "token" in wml_credentials:
                self.proceed = True
            else:
                self.proceed = False

        if 'instance_id' in wml_credentials.keys() and \
           (wml_credentials['instance_id'].lower() not in self.__predefined_instance_type_list) and \
           'version' in wml_credentials.keys():
            raise WMLClientError("Provided credentials are invalid. 'instance_id' and 'version' keys provided are not correct. Please check the wml_credentials provided."  )

        self.project_id = project_id
        self.wml_token = None

        if not self.CLOUD_PLATFORM_SPACES and not self.ICP_PLATFORM_SPACES and not self.CLOUD_BETA_FLOW:
            raise WMLClientError("This client is not supported in this release. Refer to the documentation at "
                                 "http://ibm-wml-api-pyclient.mybluemix.net to know what releases are supported")

        # if not self.ICP and not self.ICP_30 and not self.WSD:
        if not self.WSD and not self.CLOUD_PLATFORM_SPACES and not self.ICP_PLATFORM_SPACES:
            self.service_instance = ServiceInstance(self)
            self.service_instance.details = self.service_instance.get_details()

        if self.CLOUD_PLATFORM_SPACES or self.ICP_PLATFORM_SPACES:
            # For cloud, service_instance.details will be set during space creation( if instance is associated ) or
            # while patching a space with an instance
            import sys
            if (3 == sys.version_info.major) and (6 == sys.version_info.minor):
                if self.CLOUD_PLATFORM_SPACES:
                    print("DEPRECATED!! Python 3.6 framework is deprecated and will be removed on Jan 20th, 2021. "
                          "It will be read-only mode starting Nov 20th, 2020. i.e you won't be able to create new assets using this client. "
                          "Use Python 3.7 instead. For details, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html")
                elif self.ICP_PLATFORM_SPACES:
                    print("DEPRECATED!! The Python 3.6 framework is deprecated and will be removed from a future release. "
                    "We recommend you migrate your assets and use a Python 3.7 framework instead. "
                    "For details, see https://www.ibm.com/support/knowledgecenter/SSQNUZ_current/wsj/analyze-data/pm_service_supported_frameworks.html")

            self.service_instance = ServiceInstanceNewPlan(self)
            self.volumes = Volume(self)

            if self.ICP_PLATFORM_SPACES:
                self.service_instance.details = self.service_instance.get_details()

            self.set = Set(self)

            self.spaces = PlatformSpaces(self)

            self.export_assets = Export(self)
            self.import_assets = Import(self)

            if self.ICP_PLATFORM_SPACES:
                self.shiny = Shiny(self)

            self.script = Script(self)
            self.model_definitions = ModelDefinition(self)

            self.package_extensions = PkgExtn(self)
            self.software_specifications = SwSpec(self)

            self.hardware_specifications = HwSpec(self)

            self.connections = Connections(self)
            self.repository = Repository(self)
            self._functions = Functions(self)
            self.training = Training(self)

            self.data_assets = Assets(self)

            self.deployments = Deployments(self)

            if self.CLOUD_PLATFORM_SPACES:
                self.v4ga_cloud_migration = Migrationv4GACloud(self)

            self.remote_training_systems = RemoteTrainingSystem(self)

        ##Initialize Assets and Model_Definitions only for CAMS
        if (self.CAMS or self.WSD) and not self.ICP_PLATFORM_SPACES:
            self.set = Set(self)
            self.data_assets = Assets(self)
            self.model_definitions = ModelDefinition(self)
            if self.ICP_30:
                self.connections = Connections(self)
                self.software_specifications = SwSpec(self)
                self.hardware_specifications = HwSpec(self)
                self.package_extensions = PkgExtn(self)
                self.script = Script(self)
                if not '2.0' == wml_credentials[u'version'].lower():
                    self.shiny = Shiny(self)
            if self.WSD_20:
                self.software_specifications = SwSpec(self)

        #    self.learning_system = LearningSystem(self)
        self.repository = Repository(self)
        self._models = Models(self)
        self.pipelines = Pipelines(self)
        self.experiments = Experiments(self)
        self._functions = Functions(self)

        if not self.WSD and not self.CLOUD_PLATFORM_SPACES and not self.ICP_PLATFORM_SPACES:
            self.runtimes = Runtimes(self)
            self.deployments = Deployments(self)
            self.training = Training(self)
            self.spaces = Spaces(self)
            self.connections = Connections(self)
            self.experiments = Experiments(self)

        self._logger.info(u'Client successfully initialized')
        self.version = version()

    def _check_if_either_is_set(self):
        if self.CAMS or self.CLOUD_PLATFORM_SPACES:
            if self.default_space_id is None and self.default_project_id is None:
                raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

    def _check_if_space_is_set(self):
        if self.CAMS or self.CLOUD_PLATFORM_SPACES:
            if self.default_space_id is None:
                raise WMLClientError("It is mandatory to set the space. Use client.set.default_space(<SPACE_UID>) to proceed.")

    def _params(self, skip_space_project_chk=False, skip_for_create=False):
        params = {}
        if self.CAMS or self.CLOUD_PLATFORM_SPACES:
            if self.CLOUD_PLATFORM_SPACES or self.ICP_PLATFORM_SPACES:
                params.update({'version': self.version_param})
            if not skip_for_create:
                if self.default_space_id is not None:
                    params.update({'space_id': self.default_space_id})
                elif self.default_project_id is not None:
                    params.update({'project_id': self.default_project_id})
                else:
                    # For system software/hardware specs
                    if skip_space_project_chk is False:
                        raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        if self.WSD:
            if self.default_project_id is not None:
                params.update({'project_id': self.default_project_id})
            else:
                raise WMLClientError(
                    "It is mandatory to set the project id. Use client.set.default_project(<PROJECT_UID>) to proceed.")
        return params

    def _get_headers(self, content_type='application/json', no_content_type=False, wsdconnection_api_req=False, zen=False):
        if self.WSD:
                headers = {'X-WML-User-Client': 'PythonClient'}
                if self.project_id is not None:
                    headers.update({'X-Watson-Project-ID': self.project_id})
                if not no_content_type:
                    headers.update({'Content-Type': content_type})
                if wsdconnection_api_req is True:
                    token = "desktop user token"
                else:
                    token = "desktop-token"
                headers.update({'Authorization':  "Bearer " + token})

        elif zen:
            headers= {'Content-Type': content_type}
            headers.update({'Authorization': "Bearer " + self.service_instance._create_token()})
        else:
            if self.proceed is True:
                token = "Bearer "+ self.wml_credentials["token"]
            else:
                token = "Bearer " + self.service_instance._get_token()
            headers = {
                'Authorization': token,
                'X-WML-User-Client': 'PythonClient'
            }
            # Cloud Convergence
            if self._is_IAM() or (self.service_instance._is_iam() is None and not self.CLOUD_PLATFORM_SPACES and not self.ICP_PLATFORM_SPACES):
                headers['ML-Instance-ID'] = self.wml_credentials['instance_id']

            headers.update({'x-wml-internal-switch-to-new-v4': "true"})
            if not self.ICP:
                #headers.update({'x-wml-internal-switch-to-new-v4': "true"})
                if self.project_id is not None:
                    headers.update({'X-Watson-Project-ID': self.project_id})

            if not no_content_type:
                headers.update({'Content-Type': content_type})

        return headers

    def _get_icptoken(self):
        return self.service_instance._create_token()

    def _is_default_space_set(self):
        if self.default_space_id is not None:
            return True
        return False

    def _is_IAM(self):
        if('apikey' in self.wml_credentials.keys()):
            if (self.wml_credentials['apikey'] != ''):
                return True
            else:
                raise WMLClientError('apikey value cannot be \'\'. Pass a valid apikey for IAM token.')
        elif('token' in self.wml_credentials.keys()):
            if (self.wml_credentials['token'] != ''):
                return True
            else:
                raise WMLClientError('token value cannot be \'\'. Pass a valid token for IAM token.')
        else:
            return False
