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
from ibm_watson_machine_learning.utils import SPACES_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, MEMBER_DETAILS_TYPE,DATA_ASSETS_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv, str_type_conv, get_file_from_cos
from ibm_watson_machine_learning.metanames import ShinyMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os

_DEFAULT_LIST_LENGTH = 50


class Shiny(WMLResource):
    """
    Store and manage your shiny assets.

    """
    ConfigurationMetaNames = ShinyMetaNames()
    """MetaNames for Shiny Assets creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, shiny_uid):
        """
            Get shiny asset details.

            **Parameters**

            .. important::
                #. **shiny_uid**:  Unique id  of shiny asset\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: Metadata of the stored shiny asset\n
                **return type**: dict\n

            **Example**

             >>> asset_details = client.shiny.get_details(shiny_uid)

        """
        Shiny._validate_type(shiny_uid, u'asset_uid', STR_TYPE, True)


        if not self._ICP:
            response = requests.get(self._href_definitions.get_data_asset_href(shiny_uid), params=self._client._params(),
                                    headers=self._client._get_headers())
        else:
            response = requests.get(self._href_definitions.get_data_asset_href(shiny_uid), params=self._client._params(),
                                      headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            response = self._get_required_element_from_response(self._handle_response(200, u'get asset details', response))

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": {}
                }

                return final_response
            # return self._get_required_element_from_response(self._handle_response(200, u'get asset details', response))
        else:
            return self._handle_response(200, u'get asset details', response)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props, file_path):
        """
                Creates a shiny asset and uploads content to it.

                **Parameters**

                .. important::
                   #. **name**:  Name to be given to the shiny asset\n

                      **type**: str\n

                   #. **file_path**:  Path to the content file to be uploaded\n

                      **type**: str\n

                **Output**

                .. important::

                    **returns**: metadata of the stored shiny asset\n
                    **return type**: dict\n

                **Example**
                 >>> meta_props = {
                 >>>    client.shiny.ConfigurationMetaNames.NAME: "shiny app name"
                 >>> }
                 >>> asset_details = client.shiny.store(meta_props,file_path="/path/to/file")
        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        Shiny._validate_type(file_path, u'file_path', str, True)

        shiny_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        response = self._create_asset(shiny_meta, file_path)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    def _create_asset(self, shiny_meta, file_path):

        ##Step1: Create a data asset
        f = {'file': open(file_path, 'rb')}
        name = shiny_meta[u'name']

        if shiny_meta.get('description') is not None:
            desc = shiny_meta[u'description']
        else:
            desc = ""

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "shiny_asset",
                    "origin_country": "us",
                    "asset_category": "USER"
                }
            }
        else:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "shiny_asset",
                    "origin_country": "us",
                    "asset_category": "USER"
                },
                 "entity": {
                       "shiny_asset": {
                             "ml_version": "4.0.0"
                     }
                 }
            }

        base_shiny_asset = {
            "fields": [
                {
                    "key": "name",
                    "type": "string",
                    "is_array": False,
                    "is_searchable_across_types": False
                },
                {
                    "key": "entity",
                    "type": "object",
                    "is_array": False,
                    "is_searchable_across_types": False
                }
            ],
            "relationships": [],
            "name": "shiny_asset",
            "version": 1
        }

        #Step1  : Create an asset
        print("Creating Shiny asset...")

        if self._client.WSD:
            # For WSD the asset creation is done within _wsd_create_asset function using polyglot
            # Thus using the same for data_assets type


            meta_props = {
                    "name": name
            }
            details = Shiny._wsd_create_asset(self, "shiny_asset", asset_meta, meta_props, file_path, user_archive_file=True)
            return self._get_required_element_from_response(details)
        else:

            creation_response = ""
            if not self._ICP:
                creation_response = requests.post(
                        self._href_definitions.get_data_assets_href(),
                        headers=self._client._get_headers(),
                        params = self._client._params(),
                        json=asset_meta
                )
            else:
                if self._client.ICP_PLATFORM_SPACES:
                    creation_response = requests.post(
                        self._href_definitions.get_data_assets_href(),
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                        json=asset_meta,
                        verify=False
                    )
                else:
                    # Until global asset for shiny is created
                    asset_type_response = requests.post(
                        self._wml_credentials['url'] + "/v2/asset_types?",
                        headers=self._client._get_headers(),
                        json=base_shiny_asset,
                        params=self._client._params(),
                        verify=False
                    )
                    if asset_type_response.status_code == 201 or asset_type_response.status_code == 409:
                        creation_response = requests.post(
                            self._href_definitions.get_data_assets_href(),
                            headers=self._client._get_headers(),
                            json=asset_meta,
                            params=self._client._params(),
                            verify=False
                        )

            asset_details = self._handle_response(201, u'creating new asset', creation_response)
            #Step2: Create attachment
            if creation_response.status_code == 201:
                asset_id = asset_details["metadata"]["asset_id"]
                attachment_meta = {
                        "asset_type": "shiny_asset",
                        "name": "attachment_"+asset_id
                    }

                if not self._ICP:
                    attachment_response = requests.post(
                        self._href_definitions.get_attachments_href(asset_id),
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                        json=attachment_meta
                    )
                else:
                    attachment_response = requests.post(
                        self._wml_credentials['url']+"/v2/assets/"+asset_id+"/attachments",
                        headers=self._client._get_headers(),
                        json=attachment_meta,
                        params=self._client._params(),
                        verify=False
                    )
                attachment_details = self._handle_response(201, u'creating new attachment', attachment_response)
                if attachment_response.status_code == 201:
                    attachment_id = attachment_details["attachment_id"]
                    attachment_url = attachment_details["url1"]


                    #Step3: Put content to attachment
                    if not self._ICP:
                        put_response = requests.put(
                            attachment_url,
                            data=open(file_path, 'rb').read(),
                            # files = f
                        )
                    else:
                        put_response = requests.put(
                            self._wml_credentials['url'] + attachment_url,
                            files=f,
                            verify=False
                        )
                    if put_response.status_code == 201 or put_response.status_code == 200:
                        # Step4: Complete attachment
                        if not self._ICP:
                            complete_response = requests.post(
                                # self._href_definitions.get_attachment_href(asset_id,attachment_id)+"/complete",
                                self._href_definitions.get_attachment_complete_href(asset_id, attachment_id),
                                headers=self._client._get_headers(),
                                params = self._client._params()

                            )
                        else:
                            complete_response = requests.post(
                                self._href_definitions.get_attachment_href(asset_id,attachment_id)+"/complete",
                                headers=self._client._get_headers(),
                                params=self._client._params(),
                                verify=False
                            )
                        if complete_response.status_code == 200:
                            print("SUCCESS")
                            return self._get_required_element_from_response(asset_details)
                        else:
                            self._delete(asset_id)
                            raise WMLClientError("Failed while creating a shiny asset. Try again.")
                    else:
                        self._delete(asset_id)
                        raise WMLClientError("Failed while creating a shiny asset. Try again.")
                else:
                    print("SUCCESS")
                    return self._get_required_element_from_response(asset_details)
            else:
                raise WMLClientError("Failed while creating a shiny asset. Try again.")

    def list(self, limit=None):
        """
           List stored data assets. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all data assets in a table format.\n
                **return type**: None\n

           **Example**

                    >>> client.shiny.list()

        """

        Shiny._validate_type(limit, u'limit', int, False)
        href = self._href_definitions.get_search_shiny_href()

        data = {
                "query": "*:*"
        }
        if limit is not None:
            data.update({"limit": limit})

        if not self._ICP:
            response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),json=data)
        else:
            response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),json=data, verify=False)
        self._handle_response(200, u'list assets', response)
        asset_details = self._handle_response(200, u'list assets', response)["results"]
        space_values = [
            (m[u'metadata'][u'name'], m[u'metadata'][u'asset_type'], m["metadata"]["asset_id"]) for
            m in asset_details]

        self._list(space_values, [u'NAME', u'ASSET_TYPE', u'ASSET_ID'], None, _DEFAULT_LIST_LENGTH)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def download(self, asset_uid, filename, rev_uid=None):
        """
            Download the content of a shiny asset.

            **Parameters**

            .. important::
                 #. **asset_uid**:  The Unique Id of the shiny asset to be downloaded\n
                    **type**: str\n

                 #. **filename**:  filename to be used for the downloaded file\n
                    **type**: str\n

                 #. **rev_uid**:  Revision id\n
                    **type**: str\n
            **Output**

                 **returns**: Path to the downloaded asset content\n
                 **return type**: str\n

            **Example**

                  >>> client.shiny.download(asset_uid,"shiny_asset.zip")

         """
        if rev_uid is not None and self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not applicable for this release')

        Shiny._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)
        Shiny._validate_type(rev_uid, u'rev_uid', int, False)

        params = self._client._params()

        if rev_uid is not None:
            params.update({'revision_id': rev_uid})

        import urllib
        if not self._ICP:
            asset_response = requests.get(self._href_definitions.get_data_asset_href(asset_uid),
                                          params=params,
                                          headers=self._client._get_headers())
        else:
            asset_response = requests.get(self._href_definitions.get_data_asset_href(asset_uid),
                                          params=params,
                                          headers=self._client._get_headers(), verify=False)
        asset_details = self._handle_response(200, u'get assets', asset_response)

        if self._WSD:
            attachment_url = asset_details['attachments'][0]['object_key']
            artifact_content_url = self._href_definitions.get_wsd_model_attachment_href() + \
                                   urllib.parse.quote('shiny_asset/' + attachment_url, safe='')

            r = requests.get(artifact_content_url, params=self._client._params(), headers=self._client._get_headers(),
                             stream=True, verify=False)
            if r.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading data asset"), r)

            downloaded_asset = r.content
            try:
                with open(filename, 'wb') as f:
                    f.write(downloaded_asset)
                print(u'Successfully saved data asset content to file: \'{}\''.format(filename))
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(u'Saving data asset with artifact_url: \'{}\'  to local file failed.'.format(filename), e)
        else:
            attachment_id = asset_details["attachments"][0]["id"]
            if not self._ICP:
                response = requests.get(self._href_definitions.get_attachment_href(asset_uid,attachment_id), params=params,
                                        headers=self._client._get_headers())
            else:
                response = requests.get(self._href_definitions.get_attachment_href(asset_uid,attachment_id), params=params,
                                          headers=self._client._get_headers(), verify=False)
            if response.status_code == 200:
                attachment_signed_url = response.json()["url"]
                if 'connection_id' in asset_details["attachments"][0]:
                    if not self._ICP:
                        att_response = requests.get(attachment_signed_url)
                    else:
                        att_response = requests.get(attachment_signed_url,
                                                    verify=False)
                else:
                    if not self._ICP:
                        att_response = requests.get(attachment_signed_url)
                    else:
                        att_response = requests.get(self._wml_credentials["url"]+attachment_signed_url,
                                                verify=False)
                if att_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading asset"), att_response)

                downloaded_asset = att_response.content
                try:
                    with open(filename, 'wb') as f:
                        f.write(downloaded_asset)
                    print(u'Successfully saved shiny asset content to file: \'{}\''.format(filename))
                    return os.getcwd() + "/" + filename
                except IOError as e:
                    raise WMLClientError(u'Saving asset with artifact_url to local file: \'{}\' failed.'.format(filename), e)
            else:
                raise WMLClientError("Failed while downloading the asset " + asset_uid)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(asset_details):
        """
                Get Unique Id of stored shiny asset.

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the stored shiny asset\n
                      **type**: dict\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of stored shiny asset\n
                    **return type**: str\n

                **Example**

                     >>> asset_uid = client.shiny.get_id(asset_details)

        """

        return Shiny.get_uid(asset_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(asset_details):
        """
                Get Unique Id  of stored shiny asset. This method is deprecated. Use 'get_id(asset_details)' instead

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the stored shiny asset\n
                      **type**: dict\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of stored shiny asset\n
                    **return type**: str\n

                **Example**

                     >>> asset_uid = client.shiny.get_uid(asset_details)

        """
        Shiny._validate_type(asset_details, u'asset_details', object, True)
        Shiny._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(asset_details, u'data_assets_details',
                                                           [u'metadata', u'guid'])


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(asset_details):
        """
            Get url of stored shiny asset.

           **Parameters**

           .. important::
                #. **asset_details**:  stored shiny asset details\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of stored shiny asset\n
                **return type**: str\n

           **Example**

             >>> asset_details = client.shiny.get_details(asset_uid)
             >>> asset_href = client.shiny.get_href(asset_details)
        """
        Shiny._validate_type(asset_details, u'asset_details', object, True)
        Shiny._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(asset_details, u'asset_details', [u'metadata', u'href'])

    def update(self, shiny_uid, meta_props=None, file_path=None):
        """
            Update shiny with either metadata or attachment or both.

            :param shiny_uid:  Shiny UID
            :type shiny_uid: str

            :returns: updated metadata of shiny asset
            :rtype: dict

            A way you might use me is:

            >>> script_details = client.script.update(model_uid, meta, content_path)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        Shiny._validate_type(shiny_uid, 'shiny_uid', STR_TYPE, True)

        if meta_props is None and file_path is None:
            raise WMLClientError('Atleast either meta_props or file_path has to be provided')

        updated_details = None
        details = None

        url = self._href_definitions.get_asset_href(shiny_uid)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the script, delete it
        #   b. POST call to get signed url for upload
        #   c. Upload to the signed url
        #   d. Mark upload complete
        # STEP 4. Get the updated script record and return

        # STEP 1
        if not self._ICP:
            response = requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers()
            )
        else:
            response = requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers(),
                verify=False
            )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    u'Invalid input. Unable to get the details of shiny_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting shiny asset to update"), response)

        details = self._handle_response(200, "Get shiny asset details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, u'meta_props', dict, True)
            meta_props_str_conv(meta_props)

            meta_patch_payload = []

            # Since we are dealing with direct asset apis, name and description is metadata patch
            if "name" in meta_props or "description" in meta_props:
                props_for_asset_meta_patch = {}

                for key in meta_props:
                    if key == 'name' or key == 'description':
                        props_for_asset_meta_patch.update({key: meta_props[key]})

                meta_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['metadata'],
                                                                                         props_for_asset_meta_patch,
                                                                                         with_validation=True,
                                                                                         asset_meta_patch=True)
            if meta_patch_payload:
                meta_patch_url = self._href_definitions.get_asset_href(shiny_uid)

                if not self._ICP:
                    response_patch = requests.patch(meta_patch_url,
                                                    json=meta_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers())
                else:
                    response_patch = requests.patch(meta_patch_url,
                                                    json=meta_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers(), verify=False)

                updated_details = self._handle_response(200, u'shiny patch', response_patch)

        if file_path is not None:
            if "attachments" in details and details[u'attachments']:
                current_attachment_id = details[u'attachments'][0][u'id']
            else:
                current_attachment_id = None

            #STEP 3
            attachments_response = self._update_attachment_for_assets("shiny_asset",
                                                                      shiny_uid,
                                                                      file_path,
                                                                      current_attachment_id)

        if attachments_response is not None and 'success' not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._href_definitions.get_asset_href(shiny_uid)

        if not self._ICP:
            response = requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers()
            )
        else:
            response = requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers(),
                verify=False
            )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    u'Invalid input. Unable to get the details of shiny_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting shiny to update"), response)

        # response = self._handle_response(200, "Get shiny details", response)
        #
        # return self._get_required_element_from_response(response)

        response = self._get_required_element_from_response(self._handle_response(200, "Get shiny details", response))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    def _update_msg(self, updated_details):
        if updated_details is not None:
            print("Could not update the attachment because of server error."
                  " However metadata is updated. Try updating attachment again later")
        else:
            raise WMLClientError('Unable to update attachment because of server error. Try again later')


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, asset_uid):
        """
            Delete a stored shiny asset.

            **Parameters**

            .. important::
                #. **asset_uid**:  Unique Id of shiny asset\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.shiny.delete(asset_uid)

        """
        Shiny._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and \
                self._if_deployment_exist_for_asset(asset_uid):
            raise WMLClientError(
                u'Cannot delete shiny asset that has existing deployments. Please delete all associated deployments and try again')

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                    headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                      headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete assets', response)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_revision(self, shiny_uid):
        """
           Creates revision for the given Shiny asset. Revisions are immutable once created.
           The metadata and attachment at script_uid is taken and a revision is created out of it

           :param shiny_uid: Shiny asset ID. Mandatory
           :type shiny_uid: {str_type}

           :returns: stored shiny asset revisions metadata
           :rtype: dict

           >>> shiny_revision = client.shiny.create_revision(shiny_uid)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        shiny_uid = str_type_conv(shiny_uid)
        Shiny._validate_type(shiny_uid, u'shiny_uid', STR_TYPE, True)

        print("Creating shiny revision...")
        #
        # return  self._get_required_element_from_response(
        #     self._create_revision_artifact_for_assets(shiny_uid, 'Shiny'))

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(shiny_uid, 'Shiny'))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def list_revisions(self, shiny_uid, limit=None):
        """
           List all revisions for the given shiny asset uid.

           :param shiny_uid: Stored shiny asset ID.
           :type shiny_uid: {str_type}

           :param limit: limit number of fetched records (optional)
           :type limit: int

           :returns: stored shiny revisions details
           :rtype: table

           >>> client.shiny.list_revisions(shiny_uid)
        """

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        shiny_uid = str_type_conv(shiny_uid)
        Shiny._validate_type(shiny_uid, u'shiny_uid', STR_TYPE, True)

        url = self._href_definitions.get_asset_href(shiny_uid) + "/revisions"
        # /v2/assets/{asset_id}/revisions returns 'results' object
        shiny_resources = self._get_with_or_without_limit(url,
                                                           limit,
                                                           'List Shiny revisions',
                                                           summary=None,
                                                           pre_defined=None)[u'results']
        shiny_values = [
            (m[u'metadata'][u'asset_id'],
             m[u'metadata'][u'revision_id'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'commit_info'][u'committed_at']) for m in
            shiny_resources]

        self._list(shiny_values, [u'GUID', u'REVISION_ID', u'NAME', u'REVISION_COMMIT'], limit, _DEFAULT_LIST_LENGTH)

    def get_revision_details(self, shiny_uid=None, rev_uid=None):
        """
           Get metadata of shiny_uid revision.

           :param script_uid: Shiny asset ID. Mandatory
           :type script_uid: {str_type}

           :param rev_uid: Revision ID. If this parameter is not provided, returns latest revision if existing else error
           :type rev_uid: int

           :returns: stored shiny(s) metadata
           :rtype: dict

           A way you might use me is:

           >>> shiny_details = client.shiny.get_revision_details(shiny_uid, rev_uid)
        """
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        shiny_uid = str_type_conv(shiny_uid)
        Shiny._validate_type(shiny_uid, u'shiny_uid', STR_TYPE, True)
        Shiny._validate_type(rev_uid, u'rev_uid', int, False)

        if rev_uid is None:
            rev_uid = 'latest'

        url = self._href_definitions.get_asset_href(shiny_uid)
        # return self._get_required_element_from_response(self._get_with_or_without_limit(url,
        #                                        limit=None,
        #                                        op_name="asset_revision",
        #                                        summary=None,
        #                                        pre_defined=None,
        #                                        revision=rev_uid))

        response = self._get_required_element_from_response(self._get_with_or_without_limit(url,
                                               limit=None,
                                               op_name="asset_revision",
                                               summary=None,
                                               pre_defined=None,
                                               revision=rev_uid))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _delete(self, asset_uid):
        Shiny._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                       headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                       headers=self._client._get_headers(), verify=False)


    # @docstring_parameter({'str_type': STR_TYPE_NAME})
    # def get_href(self, asset_uid):
    #     """
    #        Get metadata of stored space(s). If space UID is not specified, it returns all the spaces metadata.
    #
    #        **Parameters**
    #
    #        .. important::
    #             #. **space_uid**: Space UID (optional)\n
    #                **type**: str\n
    #             #. **limit**:  limit number of fetched records (optional)\n
    #                **type**: int\n
    #
    #        **Output**
    #
    #        .. important::
    #             **returns**: metadata of stored space(s)\n
    #             **return type**: dict
    #             dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n
    #
    #        .. note::
    #             If UID is not specified, all spaces metadata is fetched\n
    #
    #        **Example**
    #
    #         >>> space_details = client.spaces.get_details(space_uid)
    #         >>> space_details = client.spaces.get_details()
    #     """
    #
    #
    #     Assets._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)
    #
    #     if not self._ICP:
    #         response = requests.get(self._href_definitions.get_data_asset_href(asset_uid), params=self._client._params(),
    #                                 headers=self._client._get_headers())
    #     else:
    #         response = requests.get(self._href_definitions.get_data_asset_href(asset_uid), params=self._client._params(),
    #                                 headers=self._client._get_headers(), verify=False)
    #     if response.status_code == 200:
    #         return response.json()["href"]
    #     else:
    #         return self._handle_response(200, u'spaces assets', response)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'shiny', dict)

        revision_id = None

        try:
            if self._client.default_space_id is not None:
                metadata = {'space_id': response_data['metadata']['space_id'],
                            'guid': response_data['metadata']['asset_id'],
                            'href': response_data['href'],
                            'name': response_data[u'metadata'][u'name'],
                            'asset_type': response_data['metadata']['asset_type'],
                            'created_at': response_data['metadata']['created_at'],
                            'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                            }
                if self._client.ICP_30 is not None or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    if "revision_id" in response_data[u'metadata']:
                        revision_id = response_data[u'metadata'][u'revision_id']
                        metadata.update({'revision_id': response_data[u'metadata'][u'revision_id']})

                    if "attachments" in response_data and response_data[u'attachments']:
                        metadata.update({'attachment_id': response_data[u'attachments'][0][u'id']})

                    if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                        metadata.update(
                            {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                new_el = {'metadata': metadata,
                          'entity': response_data['entity']
                          }

            elif self._client.default_project_id is not None:
                if self._client.WSD:

                    href = "/v2/assets/" + response_data['metadata']['asset_id'] + "?" + "project_id=" + response_data['metadata']['project_id']

                    metadata = {'project_id': response_data['metadata']['project_id'],
                                'guid': response_data['metadata']['asset_id'],
                                'href': href,
                                'name': response_data[u'metadata'][u'name'],
                                'asset_type': response_data['metadata']['asset_type'],
                                'created_at': response_data['metadata']['created_at']
                    }
                    if self._client.WSD_20 is not None:
                        if "revision_id" in response_data[u'metadata']:
                            revision_id = response_data[u'metadata'][u'revision_id']
                            metadata.update({'revision_id': response_data[u'metadata'][u'revision_id']})

                        if "attachments" in response_data and response_data[u'attachments']:
                            metadata.update({'attachment_id': response_data[u'attachments'][0][u'id']})

                        if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                            metadata.update(
                                {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
                    if 'usage' in response_data['metadata']:
                        new_el['metadata'].update(
                            {'last_updated_at': response_data['metadata']['usage']['last_updated_at']})
                    else:
                        new_el['metadata'].update(
                            {'last_updated_at': response_data['metadata']['last_updated_at']})

                else:
                    metadata = {'project_id': response_data['metadata']['project_id'],
                                'guid': response_data['metadata']['asset_id'],
                                'href': response_data['href'],
                                'name': response_data[u'metadata'][u'name'],
                                'asset_type': response_data['metadata']['asset_type'],
                                'created_at': response_data['metadata']['created_at'],
                                'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                                }

                    if self._client.ICP_30 is not None or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                        if "revision_id" in response_data[u'metadata']:
                            revision_id = response_data[u'metadata'][u'revision_id']
                            metadata.update({'revision_id': response_data[u'metadata'][u'revision_id']})

                        if "attachments" in response_data and response_data[u'attachments']:
                            metadata.update({'attachment_id': response_data[u'attachments'][0][u'id']})

                        if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                            metadata.update(
                                {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
            if 'description' in response_data['metadata']:
                new_el[u'metadata'].update({'description': response_data[u'metadata'][u'description']})

            if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'href' in response_data['metadata']:
                href_without_host = response_data['metadata']['href'].split('.com')[-1]
                new_el[u'metadata'].update({'href': href_without_host})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data.text)
