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

import requests
import json
import base64
import logging
from datetime import datetime, timedelta
from ibm_watson_machine_learning.wml_client_error import NoWMLCredentialsProvided, ApiRequestFailure, WMLClientError
from ibm_watson_machine_learning.href_definitions import HrefDefinitions


class ServiceInstanceNewPlan:
    """
        Connect, get details and check usage of your Watson Machine Learning service instance.
    """

    def __init__(self, client):
        self._logger = logging.getLogger(__name__)
        self._client = client
        self._wml_credentials = client.wml_credentials

        if self._client.ICP_PLATFORM_SPACES:
            if self.get_instance_id() == 'openshift' or self.get_instance_id() == 'wml_local':
                self._wml_credentials[u'url'] = self.get_url()
            else:
                self._wml_credentials[u'url'] = self.get_url() + ':31843'
            # TODO: Check if this is used anywhere.. from initial searches, doesn't seem like
            # self._wml_credentials[u'instance_id'] = "999"

        # This is used in connections.py
        self._href_definitions = HrefDefinitions(self._wml_credentials,
                                                 self._client.CLOUD_PLATFORM_SPACES,
                                                 self._client.PLATFORM_URL,
                                                 self._client.CAMS_URL,
                                                 self._client.ICP_PLATFORM_SPACES,
                                                 )
        self._client.wml_token = self._create_token()
        self._logger.info(u'Successfully prepared token: ' + self._client.wml_token)
        # ml_repository_client is initialized in repo
        self.details = None

    def get_instance_id(self):
        """
             Get instance id of your Watson Machine Learning service.

             **Output**

             .. important::

                **returns**: instance id\n
                **return type**: str

             **Example**

             >>> instance_details = client.service_instance.get_instance_id()
        """
        if self._wml_credentials['instance_id'] == 'invalid':
            raise WMLClientError('instance_id for this plan is picked up from the space with which'
                                 'this instance_id is associated with. Set the space with associated'
                                 'instance_id to be able to use this function')
        return self._wml_credentials['instance_id']

    def get_api_key(self):
        """
             Get api_key  of Watson Machine Learning service.
             :returns: api_key
             :rtype: str
             A way you might use me is:
             >>> instance_details = client.service_instance.get_api_key()
        """
        return self._wml_credentials['api_key']

    def get_url(self):
        """
             Get instance url of your Watson Machine Learning service.

             **Output**

             .. important::

                **returns**: instance url\n
                **return type**: str

             **Example**

             >>> instance_details = client.service_instance.get_url()
        """
        return self._wml_credentials['url']

    def get_username(self):
        """
             Get username for your Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data

             **Output**

             .. important::

                **returns**: username\n
                **return type**: str

             **Example**

             >>> instance_details = client.service_instance.get_username()
        """
        if self._client.ICP_PLATFORM_SPACES:
            return self._wml_credentials['username']
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_password(self):
        """
             Get password for your Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data

             **Output**

             .. important::

                **returns**: password\n
                **return type**: str

             **Example**

             >>> instance_details = client.service_instance.get_password()
        """
        if self._client.ICP_PLATFORM_SPACES:
            return self._wml_credentials['password']
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_details(self):
        """
             Get information about your Watson Machine Learning instance.

             **Output**

             .. important::

                **returns**: metadata of service instance\n
                **return type**: dict

             **Example**

             >>> instance_details = client.service_instance.get_details()
        """

        if not self._client.ICP:
            if self._wml_credentials is not None:

                if self._wml_credentials['instance_id'] == 'invalid':
                    raise WMLClientError('instance_id for this plan is picked up from the space with which '
                                         'this instance_id is associated with. Set the space with associated '
                                         'instance_id to be able to use this function')

                    # /ml/v4/instances will need either space_id or project_id as mandatory params
                # We will enable this service instance class only during create space or
                # set space/project. So, space_id/project_id would have been populated at this point
                headers = self._client._get_headers()

                del headers[u'X-WML-User-Client']
                if 'ML-Instance-ID' in headers:
                    headers.pop('ML-Instance-ID')
                headers.pop(u'x-wml-internal-switch-to-new-v4')
                # params = {'version': self._client.version_param}
                response_get_instance = requests.get(
                    self._href_definitions.get_v4_instance_id_href(),
                    params=self._client._params(skip_space_project_chk=True),
                    # params={'version': self._client.version_param},
                    headers=headers
                )

                if response_get_instance.status_code == 200:
                    return response_get_instance.json()
                else:
                    raise ApiRequestFailure(u'Getting instance details failed.', response_get_instance)
            else:
                raise NoWMLCredentialsProvided
        else:
            return {}


    def _get_token(self):
        if self._client.wml_token is None:
            self._create_token()
            self._client.repository._refresh_repo_client()
        else:
            if self._client.ICP:
                token_url = self._wml_credentials['url'].replace(':31002', ':31843') + '/v1/preauth/validateAuth'
                response = requests.get(token_url, auth=(self.get_username(), self.get_password()), verify=False)

                if response.status_code == 200:
                    token = response.json().get(u'accessToken')
                else:
                    raise WMLClientError(u'Error during getting token for the given credentials.', response)
                return token
            elif self._client._is_IAM():
                if self._get_expiration_datetime() - timedelta(minutes=50) < datetime.now():
                    self._client.wml_token = self._get_IAM_token()
                    self._client.repository._refresh_repo_client()
            elif self._get_expiration_datetime() - timedelta(minutes=30) < datetime.now():
                self._client.repository._refresh_repo_client()
                self._refresh_token()

        return self._client.wml_token

    def _create_token(self):

        if self._client.proceed is True:
            return self._wml_credentials["token"]

        if not self._client.ICP_PLATFORM_SPACES:
            if self._client._is_IAM():
                return self._get_IAM_token()
            else:
                raise WMLClientError('apikey for IAM token is not provided in credentials for the client.')
        else:
            token_url = self._wml_credentials['url'].replace(':31002', ':31843') + '/v1/preauth/validateAuth'
            response = requests.get(token_url, auth=(self.get_username(), self.get_password()), verify=False)

            if response.status_code == 200:
                token = response.json().get(u'accessToken')
            else:
                raise WMLClientError(u'Error during getting token for the given credentials.', response)
            return token

    def _refresh_token(self):
        import requests
        if self._client.proceed is True:
            self._client.wml_token = self._wml_credentials["token"]

        token_url = self._wml_credentials['url'].replace(':31002', ':31843') + '/v1/preauth/validateAuth'
        response = requests.get(token_url, auth=(self.get_username(), self.get_password()), verify=False)

        if response.status_code == 200:
            self._client.wml_token = response.json().get(u'accessToken')
        else:
            raise WMLClientError(u'Error during refreshing the token.', response)

    def _get_expiration_datetime(self):
        token_parts = self._client.wml_token.split('.')
        token_padded = token_parts[1] + '=' * (len(token_parts[1]) % 4)
        token_info = json.loads(base64.b64decode(token_padded).decode('utf-8'))
        token_expire = token_info.get('exp')

        return datetime.fromtimestamp(token_expire)

    def _is_iam(self):
        token_parts = self._client.wml_token.split('.')
        token_padded = token_parts[1] + '=' * (len(token_parts[1]) % 4)
        token_info = json.loads(base64.b64decode(token_padded).decode('utf-8'))
        instanceId = token_info.get('instanceId')

        return instanceId

    def _get_IAM_token(self):
        if self._client.proceed is True:
            return self._wml_credentials["token"]
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic Yng6Yng='
        }

        mystr = 'apikey=' + self._href_definitions.get_iam_token_api()
        response = requests.post(
            self._href_definitions.get_iam_token_url(),
            data=mystr,
            headers=headers
        )

        if response.status_code == 200:
            token = response.json().get(u'access_token')
        else:
            raise WMLClientError(u'Error during getting IAM Token.', response)
        return token