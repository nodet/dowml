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
from ibm_watson_machine_learning.metanames import ScriptMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os

_DEFAULT_LIST_LENGTH = 50


class Script(WMLResource):
    """
    Store and manage your scripts assets.

    """
    ConfigurationMetaNames = ScriptMetaNames()
    """MetaNames for script Assets creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, script_uid):
        """
            Get script asset details.

            **Parameters**

            .. important::
                #. **script_uid**: Unique id  of script\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: Metadata of the stored script asset\n
                **return type**: dict\n

            **Example**

             >>> script_details = client.scripts.get_details(script_uid)

        """
        Script._validate_type(script_uid, u'script_uid', STR_TYPE, True)


        if not self._ICP:
            response = requests.get(self._href_definitions.get_data_asset_href(script_uid), params=self._client._params(),
                                    headers=self._client._get_headers())
        else:
            response = requests.get(self._href_definitions.get_data_asset_href(script_uid), params=self._client._params(),
                                      headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            response = self._get_required_element_from_response(self._handle_response(200, u'get asset details', response))

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:

                entity = response[u'entity']

                try:
                    del entity[u'script'][u'ml_version']
                except KeyError:
                    pass

                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": entity
                }

                return final_response

        else:
            return self._handle_response(200, u'get asset details', response)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props, file_path):
        """
                Creates a Scripts asset and uploads content to it.

                **Parameters**

                .. important::
                   #. **meta_props**:  Name to be given to the Scripts asset\n

                      **type**: str\n

                   #. **file_path**:  Path to the content file to be uploaded\n

                      **type**: str\n

                **Output**

                .. important::

                    **returns**: metadata of the stored Scripts asset\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>>        client.script.ConfigurationMetaNamess.NAME: 'my first script',
                 >>>        client.script.ConfigurationMetaNames.DESCRIPTION: 'description of the script',
                 >>>        client.script.ConfigurationMetaNames.SOFTWARE_SPEC_UID: '0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda'
                 >>>    }
                 >>>
                 >>> asset_details = client.scripts.store(meta_props=metadata,file_path="/path/to/file")

        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        #Script._validate_type(name, u'name', str, True)
        Script._validate_type(file_path, u'file_path', str, True)
        script_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        name, extension = os.path.splitext(file_path)

        response = self._create_asset(script_meta, file_path, extension)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:

            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    def _create_asset(self, script_meta, file_path, extension='.py'):

        ##Step1: Create a data asset
        f = {'file': open(file_path, 'rb')}
        name = script_meta[u'name']

        if script_meta.get('description') is not None:
            desc = script_meta[u'description']
        else:
            desc = ""

        if script_meta.get('software_spec_uid') is not None:
            if script_meta[u'software_spec_uid'] == "":
                raise WMLClientError("Failed while creating a script asset, SOFTWARE_SPEC_UID cannot be empty")
                return

        base_script_asset = {
            "description": "Script File",
            "fields": [
                {
                    "key": "language",
                    "type": "string",
                    "facet": False,
                    "is_array": False,
                    "search_path": "asset.type",
                    "is_searchable_across_types": True
                }
            ],
            "relationships": [
                {
                    "key": "software_spec.id",
                    "target_asset_type": "software_specification",
                    "on_delete_target": "IGNORE",
                    "on_delete": "IGNORE",
                    "on_clone_target": "IGNORE"
                }
            ],
            "name": "script",
            "version": 1
        }

        if extension == '.py':
            lang = 'python3'
        elif extension == '.R':
            lang = 'R'
        else:
            raise WMLClientError("This file type is not supported. It has to be either a python script(.py file ) or a "
                                 "R script")

        #check if the software spec specified is base or derived and
        # accordingly update the entity for asset creation
        # TODO WSD2.0
        #kaganesa: remove the below if and else. once v2/software_spec apis are available in WSD2.0,
        # everything inside if should work.
        if not self._client.WSD:
            sw_spec_details = self._client.software_specifications.get_details(script_meta[u'software_spec_uid'])

            if lang == 'R':
                rscript_sw_spec_id = self._client.software_specifications.get_id_by_name('default_r3.6')

                if rscript_sw_spec_id != script_meta[u'software_spec_uid']:
                    raise WMLClientError("For R scripts, only base software spec 'default_r3.6' is supported. Specify "
                                         "the id you get via "
                                         "self._client.software_specifications.get_id_by_name('default_r3.6')")

            if(sw_spec_details[u'entity'][u'software_specification'][u'type'] == 'base'):
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    asset_meta = {
                        "metadata": {
                            "name": name,
                            "description": desc,
                            "asset_type": "script",
                            "origin_country": "us",
                            "asset_category": "USER"
                        },
                        "entity": {
                            "script": {
                                "language": {
                                    # "name": "python3"
                                    "name": lang
                    },
                                "software_spec": {
                                    "base_id": script_meta[u'software_spec_uid']
                                }
                             }
                            }
                    }
                else:
                     asset_meta = {
                         "metadata": {
                             "name": name,
                             "description": desc,
                             "asset_type": "script",
                             "origin_country": "us",
                             "asset_category": "USER"
                         },
                         "entity": {
                             "script": {
                                 "ml_version": "4.0.0",
                                 "language": {
                                     # "name": "python3"
                                     "name": lang
                     },
                                 "software_spec": {
                                     "base_id": script_meta[u'software_spec_uid']
                                 }
                             }
                         }
                     }
            else:
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    asset_meta = {
                        "metadata": {
                            "name": name,
                            "description": desc,
                            "asset_type": "script",
                            "origin_country": "us",
                            "asset_category": "USER"
                        },
                        "entity": {
                            "script": {
                                "language": {
                                    # "name": "python3"
                                    "name": lang
                    },
                                "software_spec": {
                                    "id": script_meta[u'software_spec_uid']
                                }
                            }
                        }
                    }
                else:
                    asset_meta = {
                        "metadata": {
                            "name": name,
                            "description": desc,
                            "asset_type": "script",
                            "origin_country": "us",
                            "asset_category": "USER"
                        },
                        "entity": {
                            "script": {
                                "ml_version": "4.0.0",
                                "language": {
                                    # "name": "python3"
                                    "name": lang
                                },
                                "software_spec": {
                                    "id": script_meta[u'software_spec_uid']
                                }
                            }
                        }
                    }
        else:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "script",
                    "origin_country": "us",
                    "asset_category": "USER"
                },
                "entity": {
                    "script": {
                        "language": {
                            # "name": "python3"
                            "name": lang
            },
                        "software_spec": {
                            "id": script_meta[u'software_spec_uid']
                        }
                    }
                }
            }

        #Step1  : Create an asset
        print("Creating Script asset...")

        if self._client.WSD:
            # For WSD the asset creation is done within _wsd_create_asset function using polyglot
            # Thus using the same for data_assets type


            meta_props = {
                    "name": name
            }
            details = Script._wsd_create_asset(self, "script", asset_meta, meta_props, file_path, user_archive_file=True)
            return self._get_required_element_from_response(details)
        else:
            if not self._ICP:
                creation_response = requests.post(
                        self._href_definitions.get_assets_href(),
                        headers=self._client._get_headers(),
                        params = self._client._params(),
                        json=asset_meta
                )
            else:
                if self._client.ICP_PLATFORM_SPACES:
                    creation_response = requests.post(
                        self._href_definitions.get_data_assets_href(),
                        headers=self._client._get_headers(),
                        json=asset_meta,
                        params=self._client._params(),
                        verify=False
                    )
                else:
                    asset_type_response = requests.post(
                        self._wml_credentials['url'] + "/v2/asset_types?",
                        headers=self._client._get_headers(),
                        json=base_script_asset,
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
                        "asset_type": "script",
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

                        files = open(file_path)

                        put_response = requests.put(
                            attachment_url,
                            data=files.read()
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
                            raise WMLClientError("Failed while creating a script asset. Try again.")
                    else:
                        self._delete(asset_id)
                        raise WMLClientError("Failed while creating a script asset. Try again.")
                else:
                    print("SUCCESS")
                    return self._get_required_element_from_response(asset_details)
            else:
                raise WMLClientError("Failed while creating a script asset. Try again.")


    def list(self, limit=None):
        """
           List stored scripts. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all script in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.script.list()
        """

        Script._validate_type(limit, u'limit', int, False)
        href = self._href_definitions.get_search_script_href()

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
            Download the content of a script asset.

            **Parameters**

            .. important::

                 #. **asset_uid**:  The Unique Id of the script asset to be downloaded\n
                    **type**: str\n

                 #. **filename**:  filename to be used for the downloaded file\n
                    **type**: str\n

                 #. **rev_uid**:  Revision id\n
                    **type**: str\n

            **Output**

            .. important::

                 **returns**: Path to the downloaded asset content\n
                 **return type**: str\n

            **Example**

             >>> client.script.download(asset_uid,"script_file.zip")
         """
        if rev_uid is not None and self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not applicable for this release')

        Script._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)
        Script._validate_type(rev_uid, u'rev_uid', int, False)

        params = self._client._params()

        if rev_uid is not None:
            params.update({'revision_id': rev_uid})

        import urllib
        if not self._ICP:
            asset_response = requests.get(self._href_definitions.get_asset_href(asset_uid),
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
                                   urllib.parse.quote('script/' + attachment_url, safe='')

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
                    print(u'Successfully saved data asset content to file: \'{}\''.format(filename))
                    return os.getcwd() + "/" + filename
                except IOError as e:
                    raise WMLClientError(u'Saving asset with artifact_url to local file: \'{}\' failed.'.format(filename), e)
            else:
                raise WMLClientError("Failed while downloading the asset " + asset_uid)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(asset_details):
        """
                Get Unique Id of stored script asset.

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the stored script asset\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of stored shiny asset\n
                    **return type**: str\n

                **Example**

                     >>> asset_uid = client.script.get_id(asset_details)
        """

        return Script.get_uid(asset_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(asset_details):
        """
                Get Unique Id  of stored script asset. This method is deprecated. Use 'get_id(asset_details)' instead

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the stored script asset\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of stored script asset\n
                    **return type**: str\n

                **Example**

                 >>> asset_uid = client.script.get_uid(asset_details)
        """
        Script._validate_type(asset_details, u'asset_details', object, True)
        Script._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(asset_details, u'data_assets_details',
                                                           [u'metadata', u'guid'])


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(asset_details):
        """
            Get url of stored scripts asset.

           **Parameters**

           .. important::
                #. **asset_details**:  stored script details\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of stored script asset\n
                **return type**: str\n

           **Example**

             >>> asset_details = client.script.get_details(asset_uid)
             >>> asset_href = client.script.get_href(asset_details)

        """
        Script._validate_type(asset_details, u'asset_details', object, True)
        Script._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(asset_details, u'asset_details', [u'metadata', u'href'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update(self, script_uid, meta_props=None, file_path=None):
        """
        Update script with either metadata or attachment or both.

        **Parameters**

        .. important::
            #. **param script_uid**:  Script UID.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Updated metadata of script.\n
            **return type**: dict\n

        **Example**

            >>> script_details = client.script.update(model_uid, meta, content_path)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        # We need to enable this once we add functionality for WSD
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        Script._validate_type(script_uid, 'script_uid', STR_TYPE, True)

        if meta_props is None and file_path is None:
            raise WMLClientError('Atleast either meta_props or file_path has to be provided')

        updated_details = None
        details = None

        url = self._href_definitions.get_asset_href(script_uid)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        #   b. Construct entity patch if required and call /v2/assets/<asset_id>/attributes/script to patch entity
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
                    u'Invalid input. Unable to get the details of script_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting script to update"), response)

        details = self._handle_response(200, "Get script details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, u'meta_props', dict, True)
            meta_props_str_conv(meta_props)

            meta_patch_payload = []
            entity_patch_payload = []

            # Since we are dealing with direct asset apis, there can be metadata or entity patch or both
            if "name" in meta_props or "description" in meta_props:
                props_for_asset_meta_patch = {}

                for key in meta_props:
                    if key == 'name' or key == 'description':
                        props_for_asset_meta_patch.update({key: meta_props[key]})

                meta_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['metadata'],
                                                                                         props_for_asset_meta_patch,
                                                                                         with_validation=True,
                                                                                         asset_meta_patch=True)
            # STEP 2b.
            if "software_spec_uid" in meta_props:
                if details[u'entity'][u'script'][u'software_spec']:
                    entity_patch_payload = [{'op':'replace',
                                             'path': '/software_spec/base_id',
                                             'value': meta_props[u'software_spec_uid']}]
                else:
                    entity_patch_payload = [{'op': 'add',
                                             'path': '/software_spec',
                                             'value': '{base_id:' + meta_props[u'software_spec_uid'] + '}'}]

            if meta_patch_payload:
                meta_patch_url = self._href_definitions.get_asset_href(script_uid)

                if not self._ICP:
                    response_patch = requests.patch(meta_patch_url,
                                                    json=meta_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers())
                else:
                    response_patch = requests.patch(meta_patch_url,
                                                    json=meta_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers(),verify=False)

                updated_details = self._handle_response(200, u'script patch', response_patch)

            if entity_patch_payload:
                entity_patch_url = self._href_definitions.get_asset_href(script_uid) + '/attributes/script'

                if not self._ICP:
                    response_patch = requests.patch(entity_patch_url,
                                                    json=entity_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers())
                else:
                    response_patch = requests.patch(entity_patch_url,
                                                    json=entity_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers(), verify=False)

                updated_details = self._handle_response(200, u'script patch', response_patch)

        if file_path is not None:
            if "attachments" in details and details[u'attachments']:
                current_attachment_id = details[u'attachments'][0][u'id']
            else:
                current_attachment_id = None

            #STEP 3
            attachments_response = self._update_attachment_for_assets("script",
                                                                      script_uid,
                                                                      file_path,
                                                                      current_attachment_id)

        if attachments_response is not None and 'success' not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._href_definitions.get_asset_href(script_uid)

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
                    u'Invalid input. Unable to get the details of script_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting script to update"), response)

        # response = self._handle_response(200, "Get script details", response)

        # return self._get_required_element_from_response(response)

        response = self._get_required_element_from_response(self._handle_response(200, "Get script details", response))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
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
            Delete a stored script asset.

            **Parameters**

            .. important::
                #. **asset_uid**:  Unique Id of script asset\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.script.delete(asset_uid)

        """
        Script._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and \
                self._if_deployment_exist_for_asset(asset_uid):
            raise WMLClientError(
                u'Cannot delete script that has existing deployments. Please delete all associated deployments and try again')

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
    def create_revision(self, script_uid):
        """
        Creates revision for the given script. Revisions are immutable once created.
        The metadata and attachment at script_uid is taken and a revision is created out of it

        **Parameters**

        .. important::
            #. **script_uid**: Script ID. Mandatory.\n
               **type**: str\n

        ** Output**

        .. important::

            **returns**: Stored script revisions metadata.\n
            **return type**: dict\n

        **Example**

            >>> script_revision = client.scripts.create_revision(script_uid)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        # We need to enable this once we add functionality for WSD
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        script_uid = str_type_conv(script_uid)
        Script._validate_type(script_uid, u'script_uid', STR_TYPE, True)

        print("Creating script revision...")

        # return self._get_required_element_from_response(
        #     self._create_revision_artifact_for_assets(script_uid, 'Script'))

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(script_uid, 'Script'))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def list_revisions(self, script_uid, limit=None):
        """
        List all revisions for the given script uid.

        **Parameters**

        .. important::
            #. **script_uid**: Stored script ID.\n
               **type**: str\n

            #. **limit**: Limit number of fetched records (optional).\n
               **type**: int\n

        **Output**

        .. important::
                This method only prints the list of all script in a table format.\n
                **return type**: None\n

        **Example**

            >>> client.scripts.list_revisions(script_uid)
        """
        # We need to enable this once we add functionality for WSD
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        script_uid = str_type_conv(script_uid)
        Script._validate_type(script_uid, u'script_uid', STR_TYPE, True)

        url = self._href_definitions.get_asset_href(script_uid) + "/revisions"
        # /v2/assets/{asset_id}/revisions returns 'results' object
        script_resources = self._get_with_or_without_limit(url,
                                                           limit,
                                                           'List Script revisions',
                                                           summary=None,
                                                           pre_defined=None)[u'results']
        script_values = [
            (m[u'metadata'][u'asset_id'],
             m[u'metadata'][u'revision_id'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'commit_info'][u'committed_at']) for m in
            script_resources]

        self._list(script_values, [u'GUID', u'REVISION_ID', u'NAME', u'REVISION_COMMIT'], limit, _DEFAULT_LIST_LENGTH)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_revision_details(self, script_uid=None, rev_uid=None):
        """
        Get metadata of script_uid revision.

        **Paramaters**

        .. important::
            #. **script_uid**: Script ID. Mandatory.\n
               **type**: str\n

            #. **rev_uid**: Revision ID. If this parameter is not provided, returns latest revision if existing else error.\n
               **type**: int\n

        **Output**

        .. important::

            **returns**: Stored script(s) metadata\n
            **return type**: dict\n

        **Example**

            >>> script_details = client.scripts.get_revision_details(script_uid, rev_uid)
        """
        # We need to enable this once we add functionality for WSD
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        script_uid = str_type_conv(script_uid)
        Script._validate_type(script_uid, u'script_uid', STR_TYPE, True)
        Script._validate_type(rev_uid, u'rev_uid', int, False)

        if rev_uid is None:
            rev_uid = 'latest'

        url = self._href_definitions.get_asset_href(script_uid)
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
            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _delete(self, asset_uid):
        Script._validate_type(asset_uid, u'asset_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                       headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                       headers=self._client._get_headers(), verify=False)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'scripts', dict)

        revision_id = None

        try:
            if self._client.default_space_id is not None:
                metadata = {'space_id': response_data['metadata']['space_id'],
                            'name':response_data['metadata']['name'],
                            'guid': response_data['metadata']['asset_id'],
                            'href': response_data['href'],
                            'asset_type': response_data['metadata']['asset_type'],
                            'created_at': response_data['metadata']['created_at'],
                            'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                            }
                if 'description' in response_data[u'metadata']:
                    metadata.update(
                        {'description':response_data[u'metadata'][u'description']})

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
                                'name': response_data['metadata']['name'],
                                'href': href,
                                'asset_type': response_data['metadata']['asset_type'],
                                'created_at': response_data['metadata']['created_at']
                                }

                    if 'description' in response_data[u'metadata']:
                        metadata.update(
                            {'description': response_data[u'metadata'][u'description']})

                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
                    if self._client.WSD_20 is not None:
                        if "revision_id" in response_data[u'metadata']:
                            revision_id = response_data[u'metadata'][u'revision_id']
                            new_el['metadata'].update({'revision_id': response_data[u'metadata'][u'revision_id']})

                        if "attachments" in response_data and response_data[u'attachments']:
                            new_el['metadata'].update({'attachment_id': response_data[u'attachments'][0][u'id']})

                        if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                            new_el['metadata'].update(
                                {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                    if 'usage' in response_data['metadata']:
                        new_el['metadata'].update({'last_updated_at': response_data['metadata']['usage']['last_updated_at']})
                    else:
                        new_el['metadata'].update(
                         {'last_updated_at': response_data['metadata']['last_updated_at']})
                else:
                    metadata = {'project_id': response_data['metadata']['project_id'],
                                'guid': response_data['metadata']['asset_id'],
                                'href': response_data['href'],
                                'name': response_data['metadata']['name'],
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

                    if 'description' in response_data[u'metadata']:
                        metadata.update(
                            {'description': response_data[u'metadata'][u'description']})
                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
                    if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                        href_without_host = response_data['href'].split('.com')[-1]
                        new_el[u'metadata'].update({'href': href_without_host})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data.text)
