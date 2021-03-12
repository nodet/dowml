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
import requests,json
from ibm_watson_machine_learning.utils import SPACES_IMPORTS_DETAILS_TYPE, SPACES_EXPORTS_DETAILS_TYPE, SPACES_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, MEMBER_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv, str_type_conv, get_file_from_cos, print_text_header_h2
from ibm_watson_machine_learning.metanames import SpacesMetaNames, MemberMetaNames, ExportMetaNames
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import  WMLClientError


_DEFAULT_LIST_LENGTH = 50


class Spaces(WMLResource):
    """
    Store and manage your spaces. This is applicable only for IBM Cloud Pak® for Data for Data
    """
    ConfigurationMetaNames = SpacesMetaNames()
    MemberMetaNames = MemberMetaNames()
    ExportMetaNames = ExportMetaNames()
    """MetaNames for spaces creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)


        self._ICP = client.ICP

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props):
        """
                Create a space.

                **Parameters**

                .. important::
                   #. **meta_props**:  meta data of the space configuration. To see available meta names use:\n
                                    >>> client.spaces.ConfigurationMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of the stored space\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>>  client.spaces.ConfigurationMetaNames.NAME: 'my_space',
                 >>>  client.spaces.ConfigurationMetaNames.DESCRIPTION: 'spaces',
                 >>> }
                 >>> spaces_details = client.spaces.store(meta_props=metadata)
                 >>> spaces_href = client.spaces.get_href(spaces_details)
        """

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        WMLResource._chk_and_block_create_update_for_python36(self)
        Spaces._validate_type(meta_props, u'meta_props', dict, True)
        space_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client

        )


        if not self._ICP:
            creation_response = requests.post(
                    self._wml_credentials['url'] + '/v4/spaces',
                    headers=self._client._get_headers(),
                    json=space_meta
            )
        else:
            creation_response = requests.post(
                self._wml_credentials['url'] + '/v4/spaces',
                headers=self._client._get_headers(),
                json=space_meta,
                verify=False
            )


        spaces_details = self._handle_response(201, u'creating new spaces', creation_response)

        # Cloud Convergence: Set self._client.wml_credentials['instance_id'] to instance_id
        # during client.set.default_space since that's where space is associated with client
        # and also in client.set.default_project

        return spaces_details

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(spaces_details):
        """
            Get space_href from space details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored space\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: space href\n
                **return type**: str

            **Example**

             >>> space_details = client.spaces.get_details(space_uid)
             >>> space_href = client.spaces.get_href(deployment)
        """

        Spaces._validate_type(spaces_details, u'spaces_details', object, True)
        Spaces._validate_type_of_details(spaces_details, SPACES_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(spaces_details, u'spaces_details',
                                                           [u'metadata', u'href'])

    @staticmethod
    def get_uid(spaces_details):
        """
            Get space_uid from space details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored space\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: space UID\n
                **return type**: str

            **Example**

             >>> space_details = client.spaces.get_details(space_uid)
             >>> space_uid = client.spaces.get_uid(deployment)
        """

        Spaces._validate_type(spaces_details, u'spaces_details', object, True)
        Spaces._validate_type_of_details(spaces_details, SPACES_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(spaces_details, u'spaces_details',
                                                           [u'metadata', u'guid'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, space_uid):
        """
            Delete a stored space.

            **Parameters**

            .. important::
                #. **space_uid**:  space UID\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.spaces.delete(deployment_uid)
        """

        space_uid = str_type_conv(space_uid)
        Spaces._validate_type(space_uid, u'space_uid', STR_TYPE, True)

        space_endpoint = self._href_definitions.get_space_href(space_uid)
        if not self._ICP:
            response_delete = requests.delete(space_endpoint, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(space_endpoint, headers=self._client._get_headers(), verify=False)

        return self._handle_response(204, u'space deletion', response_delete, False)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, space_uid=None, limit=None):
        """
           Get metadata of stored space(s). If space UID is not specified, it returns all the spaces metadata.

           **Parameters**

           .. important::
                #. **space_uid**: Space UID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of stored space(s)\n
                **return type**: dict
                dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

           .. note::
                If UID is not specified, all spaces metadata is fetched\n

           **Example**

            >>> space_details = client.spaces.get_details(space_uid)
            >>> space_details = client.spaces.get_details()
        """

        space_uid = str_type_conv(space_uid)
        Spaces._validate_type(space_uid, u'space_uid', STR_TYPE, False)
        Spaces._validate_type(limit, u'limit', int, False)

        href = self._href_definitions.get_spaces_href()
        if space_uid is None:
            return self._get_no_space_artifact_details(href+"?include=name,tags,custom,description", None, limit, 'spaces')
        return self._get_no_space_artifact_details(href, space_uid, limit, 'spaces')

    def list(self, limit=None):
        """
           List stored spaces. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all spaces in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.spaces.list()
        """

        space_resources = self.get_details(limit=limit)[u'resources']
        space_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at']) for m in space_resources]

        self._list(space_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update(self, space_uid, changes):
        """
                Updates existing space metadata.

                **Parameters**

                .. important::
                    #. **space_uid**:  UID of space which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of updated space\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.spaces.ConfigurationMetaNames.NAME:"updated_space"
                 >>> }
                 >>> space_details = client.spaces.update(space_uid, changes=metadata)
        """

        space_uid = str_type_conv(space_uid)
        self._validate_type(space_uid, u'space_uid', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self.get_details(space_uid)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes,
                                                                            with_validation=True)

        href = self._href_definitions.get_space_href(space_uid)
        if not self._ICP:
            response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())
        else:
            response = requests.patch(href, json=patch_payload, headers=self._client._get_headers(), verify=False)
        updated_details = self._handle_response(200, u'spaces patch', response)

        return updated_details


#######SUPPORT FOR SPACE MEMBERS

    ###GET MEMBERS DETAILS
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_members_details(self, space_uid, member_id=None, limit=None):
        """
           Get metadata of members associated with a space. If member UID is not specified, it returns all the members metadata.

           **Parameters**

           .. important::
                #. **space_uid**: member UID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of member(s) of a space\n
                **return type**: dict
                dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

           .. note::
                If member id is not specified, all members metadata is fetched\n

           **Example**

            >>> member_details = client.spaces.get_member_details(space_uid,member_id)
        """

        space_uid = str_type_conv(space_uid)
        Spaces._validate_type(space_uid, u'space_uid', STR_TYPE, True)

        member_uid = str_type_conv(member_id)
        Spaces._validate_type(member_id, u'member_id', STR_TYPE, False)

        Spaces._validate_type(limit, u'limit', int, False)

        href = self._href_definitions.get_members_href(space_uid)

        return self._get_no_space_artifact_details(href, member_uid, limit, 'space members')

    ##DELETE MEMBERS

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete_members(self, space_uid,member_id):
        """
            Delete a member associated with a space.

            **Parameters**

            .. important::
                #. **space_uid**:  space UID\n
                   **type**: str\n
                #. **member_uid**:  member UID\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.spaces.delete_member(space_uid,member_id)
        """

        space_uid = str_type_conv(space_uid)
        Spaces._validate_type(space_uid, u'space_uid', STR_TYPE, True)

        member_id = str_type_conv(member_id)
        Spaces._validate_type(member_id, u'member_id', STR_TYPE, True)

        member_endpoint = self._href_definitions.get_member_href(space_uid,member_id)
        if not self._ICP:
            response_delete = requests.delete(member_endpoint, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(member_endpoint, headers=self._client._get_headers(), verify=False)

        return self._handle_response(204, u'space member deletion', response_delete, False)

#######UPDATE MEMBERS

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update_member(self, space_uid, member_id, changes):
            """
                    Updates existing member metadata.

                    **Parameters**

                    .. important::
                        #. **space_uid**:  UID of space\n
                           **type**: str\n
                        #. **member_id**:  UID of member that needs to be updated\n
                           **type**: str\n
                        #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                           **type**: dict\n

                    **Output**

                    .. important::
                        **returns**: metadata of updated member\n
                        **return type**: dict\n

                    **Example**

                     >>> metadata = {
                     >>> client.spaces.ConfigurationMetaNames.ROLE:"viewer"
                     >>> }
                     >>> member_details = client.spaces.update_member(space_uid, member_id, changes=metadata)
            """

            space_uid = str_type_conv(space_uid)
            self._validate_type(space_uid, u'space_uid', STR_TYPE, True)
            member_id = str_type_conv(member_id)
            self._validate_type(member_id, u'member_id', STR_TYPE, True)

            self._validate_type(changes, u'changes', dict, True)
            meta_props_str_conv(changes)

            details = self.get_members_details(space_uid,member_id)

            patch_payload = self.MemberMetaNames._generate_patch_payload(details['entity'], changes,
                                                                                with_validation=True)

            href = self._href_definitions.get_member_href(space_uid,member_id)
            if not self._ICP:
                response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())
            else:
                response = requests.patch(href, json=patch_payload, headers=self._client._get_headers(), verify=False)
            updated_details = self._handle_response(200, u'members patch', response)

            return updated_details

#####CREATE MEMBER
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_member(self, space_uid,meta_props):
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
                    * client.spaces.MemberMetaNames.ROLE can be any one of the following "viewer, editor, admin"\n
                    * client.spaces.MemberMetaNames.IDENTITY_TYPE can be any one of the following "user,service"\n
                    * client.spaces.MemberMetaNames.IDENTITY can be either service-ID or IAM-userID\n

                **Example**

                 >>> metadata = {
                 >>>  client.spaces.MemberMetaNames.ROLE:"Admin",
                 >>>  client.spaces.MemberMetaNames.IDENTITY:"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                 >>>  client.spaces.MemberMetaNames.IDENTITY_TYPE:"service"
                 >>> }
                 >>> members_details = client.spaces.create_member(space_uid=space_id, meta_props=metadata)
        """

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        Spaces._validate_type(meta_props, u'meta_props', dict, True)
        space_meta = self.MemberMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client

        )


        if not self._ICP:
            creation_response = requests.post(
                    self._wml_credentials['url'] + '/v4/spaces/'+space_uid+"/members",
                    headers=self._client._get_headers(),
                    json=space_meta
            )
        else:
            creation_response = requests.post(
                self._wml_credentials['url'] + '/v4/spaces/'+space_uid+"/members",
                headers=self._client._get_headers(),
                json=space_meta,
                verify=False
            )


        members_details = self._handle_response(201, u'creating new members', creation_response)

        return members_details

    def list_members(self, space_uid ,limit=None):
            """
               List stored members of a space. If limit is set to None there will be only first 50 records shown.

               **Parameters**

               .. important::
                    #. **limit**:  limit number of fetched records\n
                       **type**: int\n

               **Output**

               .. important::
                    This method only prints the list of all members associated with a space in a table format.\n
                    **return type**: None\n

               **Example**

                >>> client.spaces.list_members()
            """

            member_resources = self.get_members_details(space_uid,limit=limit)[u'resources']
            space_values = [(m[u'metadata'][u'guid'],  m[u'entity'][u'identity'], m[u'entity'][u'identity_type'], m[u'entity'][u'role'], m[u'metadata'][u'created_at']) for m in member_resources]

            self._list(space_values, [u'GUID', u'USERNAME', u'IDENTITY_TYPE', u'ROLE', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)




    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_member_href(member_details):
        """
            Get member_href from member details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored member\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: member href\n
                **return type**: str

            **Example**

             >>> member_details = client.spaces.get_member_details(member_id)
             >>> member_href = client.spaces.get_member_href(member_details)
        """

        Spaces._validate_type(member_details, u'member details', object, True)
        Spaces._validate_type_of_details(member_details, MEMBER_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(member_details, u'member_details',
                                                           [u'metadata', u'href'])

    @staticmethod
    def get_member_uid(member_details):
        """
            Get member_uid from member details.

            **Parameters**

            .. important::
                #. **member_details**:  Metadata of the created member\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: member UID\n
                **return type**: str

            **Example**

             >>> member_details = client.spaces.get_member_details(member_id)
             >>> member_id = client.spaces.get_member_uid(member_details)
        """

        Spaces._validate_type(member_details, u'member_details', object, True)
        Spaces._validate_type_of_details(member_details, MEMBER_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(member_details, u'member_details',
                                                           [u'metadata', u'guid'])


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def imports(self, space_uid, file_path):
        """
                Updates existing space metadata.
                Imports assets in the zip file to a space

                **Parameters**

                .. important::
                    #. **space_uid**:  UID of space which definition should be updated\n
                       **type**: str\n
                    #. **file_path**:  Path to the content file to be imported\\n
                       **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of import space\n
                    **return type**: str\n

                **Example**

                 >>> space_details = client.spaces.imports(space_uid, file_path="/tmp/spaces.zip")
        """

        space_uid = str_type_conv(space_uid)
        self._validate_type(space_uid, u'space_uid', STR_TYPE, True)
        self._validate_type(file_path, u'file_path', STR_TYPE, True)



        with open(str(file_path), 'rb') as archive:
            data = archive.read()

        href = self._href_definitions.get_space_href(space_uid) + "/imports"


        if not self._ICP:
            response = requests.post(href, headers=self._client._get_headers(), data=data)
        else:
            response = requests.post(href, headers=self._client._get_headers(), data=data, verify=False)
        import_space_details = self._handle_response(202, u'spaces import', response)

        return import_space_details



    @staticmethod
    def get_imports_uid(imports_space_details):
        """
            Get imports_uid from imports space details.

            **Parameters**

            .. important::
                #. **imports_space_details**:  Metadata of the created space import\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: imports space UID\n
                **return type**: str

            **Example**

             >>> member_details = client.spaces.get_imports_details(space_uid, imports_id)
             >>> imports_id = client.spaces.get_imports_uid(imports_space_details)
        """

        Spaces._validate_type(imports_space_details, u'member_details', object, True)
        Spaces._validate_type_of_details(imports_space_details, SPACES_IMPORTS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(imports_space_details, u'imports_space_details',[u'metadata', u'guid'])

    @staticmethod
    def get_exports_uid(exports_space_details):
        """
            Get imports_uid from imports space details.

            **Parameters**

            .. important::
                #. **space_exports_details**:  Metadata of the created space import\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: exports space UID\n
                **return type**: str

            **Example**

             >>> member_details = client.spaces.get_exports_details(space_uid, exports_id)
             >>> imports_id = client.spaces.get_imports_uid(exports_space_details)
        """

        Spaces._validate_type(exports_space_details, u'exports_space_details', object, True)
        Spaces._validate_type_of_details(exports_space_details, SPACES_EXPORTS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(exports_space_details, u'exports_space_details',
                                                           [u'metadata', u'guid'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_imports_details(self, space_uid, imports_id=None, limit=None):
        """
           Get metadata of stored space(s). If space UID is not specified, it returns all the spaces metadata.

           **Parameters**

           .. important::
                #. **space_uid**: Space UID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of stored space(s)\n
                **return type**: dict
                dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

           .. note::
                If UID is not specified, all spaces metadata is fetched\n

           **Example**

            >>> space_details = client.spaces.get_imports_details(space_uid)
            >>> space_details = client.spaces.get_imports_details(space_uid,imports_id)
        """

        space_uid = str_type_conv(space_uid)
        Spaces._validate_type(space_uid, u'space_uid', STR_TYPE, False)
        Spaces._validate_type(imports_id, u'imports_uid', STR_TYPE, False)
        Spaces._validate_type(limit, u'limit', int, False)


        if imports_id is None:
            href = self._href_definitions.get_space_href(space_uid) + "/imports"
        else:
            href =  self._href_definitions.get_space_href(space_uid) + "/imports/" + imports_id

        if not self._ICP:
            response = requests.get(href, headers=self._client._get_headers())
        else:
            response = requests.get(href, headers=self._client._get_headers(), verify=False)
        import_space_details = self._handle_response(200, u'spaces import details', response)

        return import_space_details

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def exports(self, space_uid, meta_props):
        """
                Updates existing space metadata.
                exports assets in the zip file from a space

                **Parameters**

                .. important::
                   #. **meta_props**:  meta data of the space configuration. To see available meta names use:\n
                                    >>> client.spaces.ExportMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of import space\n
                    **return type**: str\n

                **Example**
                    >>> meta_props = {
                    >>>     client.spaces.ExportMetaNames.NAME: "sample",
                    >>>     client.spaces.ExportMetaNames.DESCRIPTION : "test description",
                    >>>     client.spaces.ExportMetaNames.ASSETS : {"data_assets": [], "wml_model":[]} }
                    >>> }
                    >>> space_details = client.spaces.exports(space_uid, meta_props=meta_props)

        """

        Spaces._validate_type(meta_props, u'meta_props', dict, True)
        space_exports_meta = self.ExportMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        space_exports_meta_json = json.dumps(space_exports_meta)

        space_uid = str_type_conv(space_uid)
        self._validate_type(space_uid, u'space_uid', STR_TYPE, True)

        href = self._href_definitions.get_space_href(space_uid) + "/exports"

        if not self._ICP:
            response = requests.post(href, headers=self._client._get_headers(), data=space_exports_meta_json)
        else:
            response = requests.post(href, headers=self._client._get_headers(), data=space_exports_meta_json, verify=False)
        export_space_details = self._handle_response(202, u'spaces export', response)

        return export_space_details


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_exports_details(self, space_uid, exports_id=None, limit=None):
        """
           Get details of exports for space. If exports UID is not specified, it returns all the spaces metadata.

           **Parameters**

           .. important::
                #. **space_uid**: Space UID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n
           **Output**

           .. important::
                **returns**: metadata of stored space(s)\n
                **return type**: dict
                dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

           .. note::
                If UID is not specified, all spaces metadata is fetched\n

           **Example**

            >>> space_details = client.spaces.get_exports_details(space_uid)
            >>> space_details = client.spaces.get_exports_details(space_uid,exports_id)

        """

        space_uid = str_type_conv(space_uid)
        Spaces._validate_type(space_uid, u'space_uid', STR_TYPE, False)
        Spaces._validate_type(exports_id, u'imports_uid', STR_TYPE, False)
        Spaces._validate_type(limit, u'limit', int, False)


        if exports_id is None:
            href = self._href_definitions.get_space_href(space_uid) + "/exports"
        else:
            href =  self._href_definitions.get_space_href(space_uid) + "/exports/" + exports_id

        if not self._ICP:
            response = requests.get(href, headers=self._client._get_headers())
        else:
            response = requests.get(href, headers=self._client._get_headers(), verify=False)
        export_space_details = self._handle_response(200, u'spaces exports details', response)

        return export_space_details


    def download(self, space_uid, space_exports_uid, filename=None):
        """
        Downloads zip file deployment of specified UID.

        **Parameters**

        .. important::

            #. **exports_space_uid**:  UID of virtual deployment.\n
               **type**: str\n

            #. **filename**: Filename of downloaded archive. (optional)\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Path to downloaded file.\n
            **return type**: str\n

        **Example**

         >>> client.spaces.download(space_uid)
        """

        space_exports_uid = str_type_conv(space_exports_uid)
        Spaces._validate_type(space_exports_uid, u'space_exports_uid', STR_TYPE, False)

        if space_exports_uid is not None and not is_uid(space_exports_uid):
            raise WMLClientError(u'\'space_exports_uid\' is not an uid: \'{}\''.format(space_exports_uid))

        href =  self._href_definitions.get_space_href(space_uid) + "/exports/" + space_exports_uid + "/content"

        if not self._ICP:
            response = requests.get(
                href,
                headers=self._client._get_headers()
            )
        else:
            response = requests.get(
                href,
                headers=self._client._get_headers(),
                verify=False
            )
        if filename is None:
            filename = 'wmlspace.zip'

        if response.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response.content)
                new_file.close()

                print_text_header_h2(
                    u'Successfully downloaded spaces export file: ' + str(filename))

                return filename
        else:
            raise WMLClientError(u'Unable to download spaces export: ' + response.text)

