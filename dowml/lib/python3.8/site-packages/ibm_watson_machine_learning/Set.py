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

from __future__ import print_function
import requests
from ibm_watson_machine_learning.utils import SPACES_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, MEMBER_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv, str_type_conv, get_file_from_cos
from ibm_watson_machine_learning.metanames import AssetsMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import  WMLClientError
from ibm_watson_machine_learning.instance_new_plan import ServiceInstanceNewPlan


_DEFAULT_LIST_LENGTH = 50


class Set(WMLResource):
    """
    Set a space_id/project_id to be used in the subsequent actions.
    """
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def default_space(self, space_uid):
        """
                Set a space ID.

                **Parameters**

                .. important::
                   #. **space_uid**:  GUID of the space to be used:\n

                      **type**: str\n

                **Output**

                .. important::

                    **returns**: The space that is set here is used for subsequent requests.\n
                    **return type**: str("SUCCESS"/"FAILURE")\n

                **Example**

                 >>>  client.set.default_space(space_uid)

        """
        if self._client.WSD:
            raise WMLClientError(u'Spaces API are not supported in Watson Studio Desktop.')

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            space_endpoint = self._href_definitions.get_platform_space_href(space_uid)
        else:
            space_endpoint = self._href_definitions.get_space_href(space_uid)

        if not self._ICP:
            space_details = requests.get(space_endpoint, headers=self._client._get_headers())
        else:
            space_details = requests.get(space_endpoint, headers=self._client._get_headers(), verify=False)

        if space_details.status_code == 404:
            error_msg = "Space with id '{}' does not exist".format(space_uid)
            raise WMLClientError(error_msg)
            return "FAILURE"
        elif space_details.status_code == 200:
            self._client.default_space_id = space_uid
            if  self._client.default_project_id is not None:
                print("Unsetting the project_id ...")
            self._client.default_project_id = None

            if self._client.CLOUD_PLATFORM_SPACES:
                if 'compute' in space_details.json()['entity'].keys():
                    instance_id = space_details.json()['entity']['compute'][0]['guid']
                    self._client.wml_credentials[u'instance_id'] = instance_id
                    self._client.service_instance = ServiceInstanceNewPlan(self._client)
                    self._client.service_instance.details = self._client.service_instance.get_details()

                else:
                    # Its possible that a previous space is used in the context of
                    # this client which had compute but this space doesn't have
                    self._client.wml_credentials[u'instance_id'] = 'invalid'
                    self._client.service_instance = ServiceInstanceNewPlan(self._client)
                    self._client.service_instance.details = None

            return "SUCCESS"
        else:
            print("Failed to set space.")
            print(space_details.text)
            return "FAILURE"


    ##Setting project ID
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def default_project(self, project_id):
        """
                Set a project ID.

                **Parameters**

                .. important::
                   #. **project_id**:  GUID of the project\n

                      **type**: str\n

                **Output**

                .. important::

                    **returns**: "SUCCESS"\n
                    **return type**: str\n

                **Example**

                 >>>  client.set.default_project(project_id)
        """

        if self._client.ICP and '1.1' == self._client.wml_credentials[u'version'].lower():
            raise WMLClientError(u'Project APIs are not supported in Watson Studio Local. Set space_id for the subsequent actions.')

        if self._client.ICP or self._client.WSD or self._client.CLOUD_PLATFORM_SPACES:
            if project_id is not None:
                self._client.default_project_id = project_id

                if self._client.default_space_id is not None:
                    print("Unsetting the space_id ...")
                self._client.default_space_id = None

                project_endpoint = self._href_definitions.get_project_href(project_id)
                project_details = requests.get(project_endpoint, headers=self._client._get_headers(), verify=False)
                if project_details.status_code != 200 and project_details.status_code != 204:
                    print("Failed to set Project: " + project_details.text)
                    self._client.default_project_id = None
                    return "FAILURE"
                else:
                    if self._client.CLOUD_PLATFORM_SPACES:
                        instance_id = "not_found"
                        if 'compute' in project_details.json()['entity'].keys():
                            for comp_obj in project_details.json()['entity']['compute']:
                                if comp_obj['type'] == 'machine_learning':
                                    instance_id = comp_obj['guid']
                                    break
                            self._client.wml_credentials[u'instance_id'] = instance_id
                            self._client.service_instance = ServiceInstanceNewPlan(self._client)
                            self._client.service_instance.details = self._client.service_instance.get_details()
                        else:
                            # Its possible that a previous project is used in the context of
                            # this client which had compute but this project doesn't have
                            self._client.wml_credentials[u'instance_id'] = 'invalid'
                            self._client.service_instance = ServiceInstanceNewPlan(self._client)
                            self._client.service_instance.details = None

                    return "SUCCESS"

            else:
                print("Project id can not be None. ")
                return "FAILURE"
        else:
            self._client.default_project_id = project_id

            if self._client.default_space_id is not None:
                print("Unsetting the space_id ...")
            self._client.default_space_id = None
            
            return "SUCCESS"
