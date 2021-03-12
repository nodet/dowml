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
import requests,json, time
from ibm_watson_machine_learning.utils import SPACES_IMPORTS_DETAILS_TYPE, SPACES_EXPORTS_DETAILS_TYPE, SPACES_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, MEMBER_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv, str_type_conv, get_file_from_cos, print_text_header_h2
from ibm_watson_machine_learning.utils import StatusLogger, print_text_header_h1, print_text_header_h2
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import  WMLClientError
from ibm_watson_machine_learning.metanames import SpacesPlatformMetaNames, SpacesPlatformMemberMetaNames
from ibm_watson_machine_learning.instance_new_plan import ServiceInstanceNewPlan


_DEFAULT_LIST_LENGTH = 50

class PlatformSpaces(WMLResource):
    """
    Store and manage your spaces
    """
    ConfigurationMetaNames = SpacesPlatformMetaNames()
    MemberMetaNames = SpacesPlatformMemberMetaNames()
    """MetaNames for spaces creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._client = client

    def _get_resources(self, url, op_name, params=None):
        if params is not None and 'limit' in params.keys():
            if params[u'limit'] < 1:
                raise WMLClientError('Limit cannot be lower than 1.')
            elif params[u'limit'] > 1000:
                raise WMLClientError('Limit cannot be larger than 1000.')

        if len(params) > 0:
            if not self._ICP:
                response_get = requests.get(
                    url,
                    headers=self._client._get_headers(),
                    params=params
                )
            else:
                response_get = requests.get(
                    url,
                    headers=self._client._get_headers(),
                    params=params,
                    verify=False
                )

            return self._handle_response(200, op_name, response_get)
        else:

            resources = []

            while True:
                if not self._ICP:
                    response_get = requests.get( url, headers=self._client._get_headers())
                else:
                    response_get = requests.get( url, headers=self._client._get_headers(), verify=False)

                result = self._handle_response(200, op_name, response_get)
                resources.extend(result['resources'])

                if 'next' not in result:
                    break
                else:
                    url = self._wml_credentials["url"]+result['next']['href']
                    if('start=invalid' in url):
                        break
            return {
                "resources": resources
            }

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props, background_mode=True):
        """
                Create a space. The instance associated with the space via COMPUTE will be used for billing purposes on
                cloud. Note that STORAGE and COMPUTE are applicable only for cloud

                **Parameters**

                .. important::
                    #. **meta_props**:  meta data of the space configuration. To see available meta names use:\n
                                    >>> client.spaces.ConfigurationMetaNames.get()

                      **type**: dict\n

                    #. **background_mode**:  Indicator if store() method will run in background (async) or (sync). Default: True
                      **type**: bool\n

                **Output**

                .. important::

                    **returns**: metadata of the stored space\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>>  client.spaces.ConfigurationMetaNames.NAME: 'my_space',
                 >>>  client.spaces.ConfigurationMetaNames.DESCRIPTION: 'spaces',
                 >>>  client.spaces.ConfigurationMetaNames.STORAGE: {"resource_crn": "provide crn of the COS storage"},
                 >>>  client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                 >>>                                                 "crn": "provide crn of the instance"}
                 >>> }
                 >>> spaces_details = client.spaces.store(meta_props=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        PlatformSpaces._validate_type(meta_props, u'meta_props', dict, True)

        if ('compute' in meta_props or 'storage' in meta_props) and self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' and 'COMPUTE' meta props are not applicable on "
                                 "IBM Cloud Pak® for Data. If using any of these, remove and retry")

        if 'storage' not in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' is mandatory for cloud")

        if 'compute' in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            if 'name' not in meta_props[u'compute'] or 'crn' not in meta_props[u'compute']:
                raise WMLClientError("'name' and 'crn' is mandatory for 'COMPUTE'")
            temp_meta = meta_props[u'compute']
            temp_meta.update({'type': 'machine_learning'})

            meta_props[u'compute'] = temp_meta

        space_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client

        )

        if 'compute' in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            payload_compute = []
            payload_compute.append(space_meta[u'compute'])
            space_meta[u'compute'] = payload_compute

        if not self._ICP:
            creation_response = requests.post(
                self._href_definitions.get_platform_spaces_href(),
                headers=self._client._get_headers(),
                json=space_meta)
        else:
            creation_response = requests.post(
                self._href_definitions.get_platform_spaces_href(),
                headers=self._client._get_headers(),
                json=space_meta,
                verify=False)

        spaces_details = self._handle_response(202, u'creating new spaces', creation_response)

        # Cloud Convergence: Set self._client.wml_credentials['instance_id'] to instance_id
        # during client.set.default_space since that's where space is associated with client
        # and also in client.set.default_project
        #
        if 'compute' in spaces_details['entity'].keys() and self._client.CLOUD_PLATFORM_SPACES:
            instance_id = spaces_details['entity']['compute'][0]['guid']
            self._client.wml_credentials[u'instance_id'] = instance_id
            self._client.service_instance = ServiceInstanceNewPlan(self._client)
            self._client.service_instance.details = self._client.service_instance.get_details()


        if background_mode:
            print("Space has been created. However some background setup activities might still be on-going. "
                  "Check for 'status' field in the response. It has to show 'active' before space can be used. "
                  "If its not 'active', you can monitor the state with a call to spaces.get_details(space_id)")
            return spaces_details

        else:
            # note: monitor space status
            space_id = self.get_id(spaces_details)
            print_text_header_h1(u'Synchronous space creation with id: \'{}\' started'.format(space_id))

            status = spaces_details['entity']['status'].get('state')

            with StatusLogger(status) as status_logger:
                while status not in ['failed', 'error', 'completed', 'canceled', 'active']:
                    time.sleep(10)
                    spaces_details = self.get_details(space_id)
                    status = spaces_details['entity']['status'].get('state')
                    status_logger.log_state(status)
            # --- end note

            if u'active' in status:
                print_text_header_h2(u'\nCreating space  \'{}\' finished successfully.'.format(space_id))
            else:
                raise WMLClientError(
                    f"Space {space_id} creation failed with status: {spaces_details['entity']['status']}")

            return spaces_details

    @staticmethod
    def get_id(space_details):
        """
            Get space_id from space details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored space\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: space ID\n
                **return type**: str

            **Example**

             >>> space_details = client.spaces.store(meta_props)
             >>> space_id = client.spaces.get_id(space_details)
        """

        PlatformSpaces._validate_type(space_details, u'space_details', object, True)

        return WMLResource._get_required_element_from_dict(space_details, u'space_details',
                                                           [u'metadata', u'id'])


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(space_details):
        """
                Get Unique Id of the space. This method is deprecated. Use 'get_id(space_details)' instead

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the space\n
                      **type**: dict\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of space\n
                    **return type**: str\n

                **Example**

                 >>> space_details = client.spaces.store(meta_props)
                 >>> space_uid = client.spaces.get_uid(space_details)

        """
        PlatformSpaces._validate_type(space_details, u'space_details', object, True)

        return WMLResource._get_required_element_from_dict(space_details, u'space_details',
                                                           [u'metadata', u'id'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, space_id):
        """
            Delete a stored space.

            **Parameters**

            .. important::
                #. **space_uid**:  space ID\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.spaces.delete(deployment_id)
        """

        space_id = str_type_conv(space_id)
        PlatformSpaces._validate_type(space_id, u'space_id', STR_TYPE, True)

        space_endpoint = self._href_definitions.get_platform_space_href(space_id)

        if not self._ICP:
            response_delete = requests.delete(space_endpoint, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(space_endpoint, headers=self._client._get_headers(), verify=False)

        response = self._handle_response(202, u'space deletion', response_delete, False)

        print('DELETED')

        return response

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, space_id=None, limit=None):
        """
           Get metadata of stored space(s)

           **Parameters**

           .. important::
                #. **space_id**: Space ID \n
                   **type**: str\n
                #. **limit**: Applicable when space_id is not provided. If space_id is provided, this will be ignored \n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: metadata of stored space(s)\n
                **return type**: dict

           **Example**

            >>> space_details = client.spaces.get_details(space_uid)
        """

        space_id = str_type_conv(space_id)
        # PlatformSpaces._validate_type(space_id, u'space_id', STR_TYPE, True)
        PlatformSpaces._validate_type(space_id, u'space_id', STR_TYPE, False)

        href = self._href_definitions.get_platform_space_href(space_id)

        if space_id is not None:
            if not self._ICP:
                response_get = requests.get(href, headers=self._client._get_headers())
            else:
                response_get = requests.get(href, headers=self._client._get_headers(), verify=False)

            return self._handle_response(200, 'Get space', response_get)

        else:
            return self._get_with_or_without_limit(self._href_definitions.get_platform_spaces_href(),
                                                   limit,
                                                   'spaces',
                                                   summary=False,
                                                   pre_defined=False,
                                                   skip_space_project_chk=True)

    def list(self, limit=None, member=None, roles=None):
        """
           List stored spaces. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n
                #. **member**:  Filters the result list to only include spaces where the user with a matching user id
                                is a member\n
                   **type**: string\n
                #. **roles**:  limit number of fetched records\n
                   **type**: string\n

           **Output**

           .. important::
                This method only prints the list of all spaces in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.spaces.list()
        """

        PlatformSpaces._validate_type(limit, u'limit', int, False)
        href = self._href_definitions.get_platform_spaces_href()

        params = {}

        if limit is not None:
            params.update({'limit': limit})

        if limit is None:
            params.update({'limit': 50})

        if member is not None:
            params.update({'member': member})

        if roles is not None:
            params.update({'roles': roles})

        space_resources = self._get_resources(href, 'spaces', params)[u'resources']

        # space_resources = self._get_no_space_artifact_details(href, None, limit, 'spaces')[u'resources']

        space_values = [(m[u'metadata'][u'id'],
                         m[u'entity'][u'name'],
                         m[u'metadata'][u'created_at']) for m in space_resources]

        if limit is None:
            print("Note: 'limit' is not provided. Only first 50 records will be displayed if the number of records "
                  "exceed 50")

        self._list(space_values, [u'ID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update(self, space_id, changes):
        """
                Updates existing space metadata. 'STORAGE' cannot be updated
                STORAGE and COMPUTE are applicable only for cloud

                **Parameters**

                .. important::
                    #. **space_uid**:  ID of space which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of updated space\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.spaces.ConfigurationMetaNames.NAME:"updated_space",
                 >>> client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                 >>>                                                "crn": "v1:staging:public:pm-20-dev:us-south:a/09796a1b4cddfcc9f7fe17824a68a0f8:f1026e4b-77cf-4703-843d-c9984eac7272::"
                 >>>                                               }
                 >>> }
                 >>> space_details = client.spaces.update(space_id, changes=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if ('compute' in changes or 'storage' in changes) and self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' and 'COMPUTE' meta props are not applicable on"
                                 "IBM Cloud Pak® for Data. If using any of these, remove and retry")

        if 'storage' in changes:
            raise WMLClientError("STORAGE cannot be updated")

        space_id = str_type_conv(space_id)
        self._validate_type(space_id, u'space_id', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self.get_details(space_id)

        if 'compute' in changes and self._client.CLOUD_PLATFORM_SPACES:
            changes[u'compute'][u'type'] = 'machine_learning'

            payload_compute = []
            payload_compute.append(changes[u'compute'])
            changes[u'compute'] = payload_compute

        print("changes in update: ", changes)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes)

        print("patch payload: ", patch_payload)

        href = self._href_definitions.get_platform_space_href(space_id)

        if not self._ICP:
            response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())
        else:
            response = requests.patch(href, json=patch_payload, headers=self._client._get_headers(), verify=False)

        updated_details = self._handle_response(200, u'spaces patch', response)

        # Cloud Convergence
        if 'compute' in updated_details['entity'].keys() and self._client.CLOUD_PLATFORM_SPACES:
            instance_id = updated_details['entity']['compute'][0]['guid']
            self._client.wml_credentials[u'instance_id'] = instance_id
            self._client.service_instance = ServiceInstanceNewPlan(self._client)
            self._client.service_instance.details = self._client.service_instance.get_details()

        return updated_details


#######SUPPORT FOR SPACE MEMBERS

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_member(self, space_id, meta_props):
        """
                Create a member within a space.

                **Parameters**

                .. important::
                   #. **meta_props**:  meta data of the member configuration. To see available meta names use:\n
                                    >>> client.spaces.MemberMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of the stored member\n
                    **return type**: dict\n

                .. note::
                    * 'role' can be any one of the following "viewer, editor, admin"\n
                    * 'type' can be any one of the following "user,service"\n
                    * 'id' can be either service-ID or IAM-userID\n

                **Example**

                 >>> metadata = {
                 >>>  client.spaces.MemberMetaNames.MEMBERS: [{"id":"IBMid-100000DK0B", "type": "user", "role": "admin" }]
                 >>> }
                 >>> members_details = client.spaces.create_member(space_id=space_id, meta_props=metadata)

                 >>> metadata = {
                 >>>  client.spaces.MemberMetaNames.MEMBERS: [{"id":"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71", "type": "service", "role": "admin" }]
                 >>> }
                 >>> members_details = client.spaces.create_member(space_id=space_id, meta_props=metadata)
        """

        space_id = str_type_conv(space_id)
        self._validate_type(space_id, u'space_id', STR_TYPE, True)

        PlatformSpaces._validate_type(meta_props, u'meta_props', dict, True)

        meta = {}

        if 'members' in meta_props:
            meta = meta_props
        elif 'member' in meta_props:
            dictionary = meta_props['member']
            payload = []
            payload.append(dictionary)
            meta['members'] = payload

        space_meta = self.MemberMetaNames._generate_resource_metadata(
            meta,
            with_validation=True,
            client=self._client
        )

        if not self._ICP:
            creation_response = requests.post(
                self._href_definitions.get_platform_spaces_members_href(space_id),
                headers=self._client._get_headers(),
                json=space_meta)
        else:
            creation_response = requests.post(
                self._href_definitions.get_platform_spaces_members_href(space_id),
                headers=self._client._get_headers(),
                json=space_meta,
                verify=False)

        # TODO: Change response code one they change it to 201
        members_details = self._handle_response(200, u'creating new members', creation_response)

        return members_details

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_member_details(self, space_id, member_id):
        """
           Get metadata of member associated with a space

           **Parameters**

           .. important::
                #. **space_id**: member ID \n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: metadata of member of a space\n
                **return type**: dict

           **Example**

            >>> member_details = client.spaces.get_member_details(space_uid,member_id)
        """

        space_id = str_type_conv(space_id)
        PlatformSpaces._validate_type(space_id, u'space_id', STR_TYPE, True)

        member_id = str_type_conv(member_id)
        PlatformSpaces._validate_type(member_id, u'member_id', STR_TYPE, True)

        href = self._href_definitions.get_platform_spaces_member_href(space_id, member_id)

        if not self._ICP:
            response_get = requests.get(href, headers=self._client._get_headers())
        else:
            response_get = requests.get(href, headers=self._client._get_headers(), verify=False)

        return self._handle_response(200, 'Get space member', response_get)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete_member(self, space_id, member_id):
        """
            Delete a member associated with a space.

            **Parameters**

            .. important::
                #. **space_id**:  space UID\n
                   **type**: str\n
                #. **member_id**:  member UID\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.spaces.delete_member(space_id,member_id)
        """

        space_id = str_type_conv(space_id)
        PlatformSpaces._validate_type(space_id, u'space_id', STR_TYPE, True)

        member_id = str_type_conv(member_id)
        PlatformSpaces._validate_type(member_id, u'member_id', STR_TYPE, True)

        member_endpoint = self._href_definitions.get_platform_spaces_member_href(space_id, member_id)

        if not self._ICP:
            response_delete = requests.delete(member_endpoint, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(member_endpoint, headers=self._client._get_headers(), verify=False)

        print('DELETED')

        return self._handle_response(204, u'space member deletion', response_delete, False)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update_member(self, space_id, member_id, changes):
            """
                    Updates existing member metadata.

                    **Parameters**

                    .. important::
                        #. **space_id**:  ID of space\n
                           **type**: str\n
                        #. **member_id**:  ID of member that needs to be updated\n
                           **type**: str\n
                        #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                           **type**: dict\n

                    **Output**

                    .. important::
                        **returns**: metadata of updated member\n
                        **return type**: dict\n

                    **Example**

                     >>> metadata = {
                     >>>  client.spaces.MemberMetaNames.MEMBER: {"role": "editor"}
                     >>> }
                     >>> member_details = client.spaces.update_member(space_id, member_id, changes=metadata)
            """

            space_id = str_type_conv(space_id)
            self._validate_type(space_id, u'space_id', STR_TYPE, True)
            member_id = str_type_conv(member_id)
            self._validate_type(member_id, u'member_id', STR_TYPE, True)

            self._validate_type(changes, u'changes', dict, True)
            meta_props_str_conv(changes)

            details = self.get_member_details(space_id, member_id)

            # The member record is a bit different than most other type of records we deal w.r.t patch
            # There is no encapsulating object for the fields. We need to be consistent with the way we
            # provide the meta in create/patch. When we give with .MEMBER, _generate_patch_payload
            # will generate with /member patch. So, separate logic for member patch inline here
            changes1 = changes['member']

            # Union of two dictionaries. The one in changes1 will override existent ones in current meta
            details.update(changes1)

            id_str = {}
            role_str = {}
            type_str = {}
            state_str = {}

            # if 'id' in details:
            #     id_str["op"] = "replace"
            #     id_str["path"] = "/id"
            #     id_str["value"] = details[u'id']
            if 'role' in details:
                role_str["op"] = "replace"
                role_str["path"] = "/role"
                role_str["value"] = details[u'role']
            # if 'type' in details:
            #     type_str["op"] = "replace"
            #     type_str["path"] = "/type"
            #     type_str["value"] = details[u'type']
            if 'state' in details:
                state_str["op"] = "replace"
                state_str["path"] = "/state"
                state_str["value"] = details[u'state']

            patch_payload = []

            # if id_str:
            #     patch_payload.append(id_str)
            if role_str:
                patch_payload.append(role_str)
            # if type_str:
            #     patch_payload.append(type_str)
            if state_str:
                patch_payload.append(state_str)

             # patch_payload = self.MemberMetaNames._generate_patch_payload(details, changes, with_validation=True)

            href = self._href_definitions.get_platform_spaces_member_href(space_id,member_id)

            if not self._ICP:
                response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())
            else:
                response = requests.patch(href, json=patch_payload, headers=self._client._get_headers(), verify=False)

            updated_details = self._handle_response(200, u'members patch', response)

            return updated_details

    def list_members(self, space_id, limit=None, identity_type=None, role=None, state=None):
            """
               List stored members of a space. If limit is set to None there will be only first 50 records shown.

               **Parameters**

               .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n
                #. **identity_type**:  Find the member by type\n
                   **type**: string\n
                #. **role**:  Find the member by role\n
                   **type**: string\n
                #. **state**:  Find the member by state\n
                   **type**: string\n

               **Output**

               .. important::
                    This method only prints the list of all members associated with a space in a table format.\n
                    **return type**: None\n

               **Example**

                >>> client.spaces.list_members(space_id)
            """

            space_id = str_type_conv(space_id)
            self._validate_type(space_id, u'space_id', STR_TYPE, True)

            params = {}

            if limit is not None:
                params.update({'limit': limit})

            if limit is None:
                params.update({'limit': 50})

            if identity_type is not None:
                params.update({'type': identity_type})

            if role is not None:
                params.update({'role': role})

            if state is not None:
                params.update({'state': state})

            href = self._href_definitions.get_platform_spaces_members_href(space_id)

            member_resources = self._get_resources(href, 'space members', params)[u'resources']

            # space_values = [(m[u'metadata'][u'id'],
            #                  m[u'entity'][u'id'],
            #                  m[u'entity'][u'type'],
            #                  m[u'entity'][u'role'],
            #                  m[u'entity'][u'state'],
            #                  m[u'metadata'][u'created_at']) for m in member_resources]


            # self._list(space_values, [u'ID', u'IDENTITY',
            #                           u'IDENTITY_TYPE',
            #                           u'ROLE',
            #                           u'STATE',
            #                           u'CREATED'], limit, _DEFAULT_LIST_LENGTH)

            space_values = [(m[u'id'],
                             m[u'type'],
                             m[u'role'],
                             m[u'state']) if 'state' in m else
                            (m[u'id'],
                             m[u'type'],
                             m[u'role'],
                             None) for m in member_resources]

            if limit is None:
                print("Note: 'limit' is not provided. Only first 50 records will be displayed if the number of records "
                      "exceed 50")

            self._list(space_values, [u'ID',
                                      u'TYPE',
                                      u'ROLE',
                                      u'STATE'], limit, _DEFAULT_LIST_LENGTH)





