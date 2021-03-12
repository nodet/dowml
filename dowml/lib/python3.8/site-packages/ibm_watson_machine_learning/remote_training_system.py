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
from ibm_watson_machine_learning.utils import STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv
from ibm_watson_machine_learning.metanames import RemoteTrainingSystemMetaNames
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
_DEFAULT_LIST_LENGTH = 50

class RemoteTrainingSystem(WMLResource):
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._client = client
        self.ConfigurationMetaNames = RemoteTrainingSystemMetaNames()

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props):
        """
            Create a remote training system. Either space_id or project_id has to be provided and is mandatory.

           **Parameters**

           .. important::
                #. **meta_props**:  meta data. To see available meta names use **client.remote_training_systems.ConfigurationMetaNames.get()**\n
                   **type**: str or dict\n

           **Output**

           .. important::
                **returns**: Response json\n
                **return type**: dict\n

           **Example**

                >>> metadata = {
                >>>    client.remote_training_systems.ConfigurationMetaNames.NAME: "my-resource",
                >>>    client.remote_training_systems.ConfigurationMetaNames.TAGS: ["tag1", "tag2"],
                >>>    client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: {"name": "name", "region": "EU"}
                >>>    client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES: [{"id": "43689024", "type": "user"}],
                >>>    client.remote_training_systems.ConfigurationMetaNames.REMOTE_ADMIN: {"name": "name", "email": "email@email.com"}
                >>> }
                >>> client.set.default_space('3fc54cf1-252f-424b-b52d-5cdd9814987f')
                >>> details = client.remote_training_systems.store(meta_props=metadata)
            """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")
        
        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(meta_props, u'meta_props', dict, True)
        self._validate_input(meta_props)

        meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        if self._client.default_space_id is not None:
            meta['space_id'] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            meta['project_id'] = self._client.default_project_id

        href = self._href_definitions.remote_training_systems_href()

        if not self._ICP:
            creation_response = requests.post(href,
                                              params=self._client._params(),
                                              headers=self._client._get_headers(),
                                              json=meta)
        else:
            creation_response = requests.post(href,
                                              params=self._client._params(),
                                              headers=self._client._get_headers(),
                                              json=meta,
                                              verify=False)

        details = self._handle_response(expected_status_code=201,
                                        operationName=u'store remote training system specification',
                                        response=creation_response)

        return details

    def _validate_input(self, meta_props):
        if 'name' not in meta_props:
            raise WMLClientError("Its mandatory to provide 'NAME' in meta_props. Example: "
                                 "client.remote_training_systems.ConfigurationMetaNames.NAME")

        if 'allowed_identities' not in meta_props:
            raise WMLClientError("Its mandatory to provide 'ALLOWED_IDENTITIES' in meta_props. Example: "
                                 "client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES")

        if 'organization' in meta_props and 'name' not in meta_props[u'organization']:
            raise WMLClientError("Its mandatory to provide 'name' for ORGANIZATION meta_prop. Eg: "
                                 "client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: "
                                 "{'name': 'org'} ")


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, remote_training_systems_id):
        """
            Deletes the given remote_training_systems_id definition. 'space_id' or 'project_id' has to be provided

            **Parameters**

            .. important::
                #. **remote_training_systems_id**:  Remote Training System identifier\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.remote_training_systems.delete(remote_training_systems_id='6213cf1-252f-424b-b52d-5cdd9814956c')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        self._client._check_if_either_is_set()
        RemoteTrainingSystem._validate_type(remote_training_systems_id, u'remote_training_systems_id', STR_TYPE, True)

        href = self._href_definitions.remote_training_system_href(remote_training_systems_id)

        if not self._ICP:
            delete_response = requests.delete(href,
                                              params=self._client._params(),
                                              headers=self._client._get_headers())
        else:
            delete_response = requests.delete(href,
                                              params=self._client._params(),
                                              headers=self._client._get_headers(),
                                              verify=False)

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'delete remote training system definition',
                                        response=delete_response)

        if "SUCCESS" == details:
            print("Remote training system deleted")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, remote_training_system_id=None, limit=None):
        """
            Get metadata of the given remote training system. if no remote_training_systems_id is specified all 
            remote training systems metadata is returned.

           **Parameters**

           .. important::
                #. **remote_training_system_id**: remote training system identifier (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: remote training system(s) metadata\n
                **return type**: dict\n
                dict (if remote_training_systems_id is not None) or {"resources": [dict]} (if remote_training_systems_id is None)\n

           .. note::
                If remote_training_systems_id is not specified, all remote training system(s) metadata is fetched\n

           **Example**

             >>> details = client.remote_training_systems.get_details(remote_training_systems_id)
             >>> details = client.remote_training_systems.get_details()
         """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")
        
        self._client._check_if_either_is_set()
        
        RemoteTrainingSystem._validate_type(remote_training_system_id, u'remote_training_systems_id', STR_TYPE, False)
        RemoteTrainingSystem._validate_type(limit, u'limit', int, False)

        href = self._href_definitions.remote_training_systems_href()

        return self._get_artifact_details(href, remote_training_system_id, limit, 'remote_training_systems')
    
    def list(self, limit=None):
        """
            List stored remote training systems. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all remote training systems in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.remote_training_systems.list()
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        self._client._check_if_either_is_set()

        resources = self.get_details()[u'resources']

        values = [(m[u'metadata'][u'id'],
                   m[u'metadata'][u'name'],
                   m[u'metadata'][u'created_at']) for m in resources]

        self._list(values, [u'ID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(remote_training_system_details):
        """
                Get ID of remote_training_system

                **Parameters**

                .. important::

                   #. **remote_training_system_details**:  Metadata of the stored remote training system\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: ID of stored remote training system\n
                    **return type**: str\n

                **Example**

                 >>> details = client.remote_training_systems.get_details(remote_training_system_id)
                 >>> id = client.remote_training_systems.get_id(details)
        """
        RemoteTrainingSystem._validate_type(remote_training_system_details, u'remote_training_system_details', object, True)

        return WMLResource._get_required_element_from_dict(remote_training_system_details,
                                                           u'remote_training_system_details',
                                                           [u'metadata', u'id'])

    def update(self, remote_training_system_id, changes):
        """
        Updates existing remote training system metadata.

        **Parameters**

        .. important::
           #. **remote_training_system_id**:  remote training system identifier\n
              **type**: str\n
           #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
              **type**: dict\n

        **Example**
         >>> metadata = {
         >>> client.remote_training_systems.ConfigurationMetaNames.NAME:"updated_remote_training_system"
         >>> }
         >>> details = client.remote_training_systems.update(remote_training_system_id, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        self._client._check_if_either_is_set()

        self._validate_type(remote_training_system_id, u'remote_training_system_id', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self.get_details(remote_training_system_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes,
                                                                            with_validation=True)

        href = self._href_definitions.remote_training_system_href(remote_training_system_id)

        if not self._ICP:
            response = requests.patch(href,
                                      json=patch_payload,
                                      params = self._client._params(),
                                      headers=self._client._get_headers())
        else:
            response = requests.patch(href,
                                      json=patch_payload,
                                      params = self._client._params(),
                                      headers=self._client._get_headers(),
                                      verify=False)

        updated_details = self._handle_response(200, u'remote training system patch', response)

        return updated_details

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_revision(self, remote_training_system_id):
        """
            Create a new remote training system revision.

            :param remote_training_system_id: Unique remote training system ID
            :type remote_training_system_id: {str_type}

            Example:

            >>> client.remote_training_systems.create_revision(remote_training_system_id)
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        RemoteTrainingSystem._validate_type(remote_training_system_id, u'remote_training_system_id', STR_TYPE, False)

        href = self._href_definitions.remote_training_systems_href()
        return self._create_revision_artifact(href, remote_training_system_id, 'remote training system')

    def get_revision_details(self, remote_training_system_id, rev_id):
        """
           Get metadata of specific revision of stored remote system

           :param remote_training_system_id: stored functions, definition
           :type remote_training_system_id: {str_type}

           :param rev_id: Unique id of the remote system revision.
           :type rev_id : int

           :returns: stored remote system revision metadata
           :rtype: dict

           Example:

           >>> details = client.remote_training_systems.get_details(remote_training_system_id, rev_id)
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(remote_training_system_id, u'remote_training_system_id', STR_TYPE, True)
        RemoteTrainingSystem._validate_type(rev_id, u'rev_id', int, True)

        href = self._href_definitions.remote_training_system_href(remote_training_system_id)
        return self._get_with_or_without_limit(href, limit=None, op_name="remote_training_system_id",
                                               summary=None, pre_defined=None, revision=rev_id)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def list_revisions(self, remote_training_system_id, limit=None):
        """
           List all revisions for the given remote_training_system_id

           :param remote_training_system_id: Unique id of stored remote system
           :type remote_training_system_id: {str_type}

           :param limit: limit number of fetched records (optional)
           :type limit: int

           :returns: list all remote systems revisions summary.
           :rtype: table

           >>> details = client.remote_training_systems.list_revisions(remote_training_system_id)
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")
        
        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(remote_training_system_id, u'remote_training_system_id', STR_TYPE, True)

        href = self._href_definitions.get_function_href(remote_training_system_id)

        resources = self._get_artifact_details(href + '/revisions',
                                               None,
                                               limit,
                                               'remote system revisions')[u'resources']

        values = [(m[u'metadata'][u'id'],
             m[u'metadata'][u'rev'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'created_at']) for m in resources]

        self._list(values, [u'ID', u'rev', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
