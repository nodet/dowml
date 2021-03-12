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
from ibm_watson_machine_learning.metanames import ModelDefinitionMetaNames, MetaNamesBase,  MetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError
import os
import json
import uuid

_DEFAULT_LIST_LENGTH = 50


class ModelDefinition(WMLResource):
    """
    Store and manage your model_definitions.
    """

    ConfigurationMetaNames = ModelDefinitionMetaNames()

    """MetaNames for model_definition creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP
        self.default_space_id = client.default_space_id

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _generate_model_definition_document(self, meta_props):
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            doc = {
                "metadata":
                {
                   "name": "generated_name_"+str(uuid.uuid4()),
                   "tags": ["generated_tag_"+str(uuid.uuid4())],
                   "asset_type": "wml_model_definition",
                   "origin_country": "us",
                   "rov": {
                      "mode": 0
                   },
                   "asset_category": "USER"
                },
                "entity": {
                   "wml_model_definition": {
                       "ml_version": "4.0.0",
                       "version": "1.0",
                       "platform": {
                          "name": "python",
                          "versions": [
                              "3.5"
                             ]
                       }
                   }
                }
            }

        else:
            doc = {
                "metadata":
                {
                   "name": "My wml_model_definition assert",
                   "tags": ["string"],
                   "asset_type": "wml_model_definition",
                   "origin_country": "us",
                   "rov": {
                      "mode": 0
                   },
                   "asset_category": "USER"
                },
                "entity": {
                   "wml_model_definition": {
                       "name": "tf-model_trainings_v4_test_suite_basic",
                       "description": "Sample custom library",
                       "version": "1.0",
                       "platform": {
                          "name": "python",
                          "versions": [
                              "3.5"
                             ]
                       }
                   }
                }
            }

        if self.ConfigurationMetaNames.NAME in meta_props:
            doc["metadata"]["name"] = meta_props[self.ConfigurationMetaNames.NAME]
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                # We shouldn't have name and description in entity but this code exists in CP4D
                # So, changing only for Cloud Convergence
                doc["entity"]["wml_model_definition"]["name"] = meta_props[self.ConfigurationMetaNames.NAME]
        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            doc["metadata"]["description"] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                doc["entity"]["wml_model_definition"]["description"] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

        if self.ConfigurationMetaNames.VERSION in meta_props:
            doc["entity"]["wml_model_definition"]["version"] = meta_props[self.ConfigurationMetaNames.VERSION]

        if self.ConfigurationMetaNames.PLATFORM in meta_props:
            doc["entity"]["wml_model_definition"]["platform"]["name"] = meta_props[self.ConfigurationMetaNames.PLATFORM]['name']
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                doc["entity"]["wml_model_definition"]["platform"]["versions"] = \
                    meta_props[self.ConfigurationMetaNames.PLATFORM]['versions']
            else:
                doc["entity"]["wml_model_definition"]["platform"]["versions"][0] = \
                    meta_props[self.ConfigurationMetaNames.PLATFORM]['versions'][0]

        if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            if self.ConfigurationMetaNames.COMMAND in meta_props:
                doc['entity']['wml_model_definition'].update({"command": meta_props[self.ConfigurationMetaNames.COMMAND]})
            if self.ConfigurationMetaNames.CUSTOM in meta_props:
                doc['entity']['wml_model_definition'].update(
                    {"custom": meta_props[self.ConfigurationMetaNames.CUSTOM]})

        return doc

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, model_definition, meta_props):
        """
        Create a model_definitions.\n

        **Parameters**

        .. important::

            #. **meta_props**:  meta data of the model_definition configuration. To see available meta names use:\n
                >>> client.model_definitions.ConfigurationMetaNames.get()
               **type**: dict\n

            #. **model_definition**:  Path to the content file to be uploaded\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Metadata of the model_defintion created\n
            **return type**: dict\n

        **Example**

            >>>  client.model_definitions.store(model_definition, meta_props)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        model_definition = str_type_conv(model_definition)
        self.ConfigurationMetaNames._validate(meta_props)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:

            # metadata doesn't seem to be used at all.. Code existing pre-convergence.
            # Should be checked sometime and removed if not needed
            metadata = {
                self.ConfigurationMetaNames.NAME: meta_props[self.ConfigurationMetaNames.NAME],
                self.ConfigurationMetaNames.VERSION: meta_props['version'],
                self.ConfigurationMetaNames.PLATFORM:
                    meta_props['platform']
                    if 'platform' in meta_props and meta_props['platform'] is not None
                    else {
                        "name": meta_props[self.ConfigurationMetaNames.PLATFORM]['name'],
                        "versions": [meta_props[self.ConfigurationMetaNames.PLATFORM]['versions']]
                    },
            }

            if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
                metadata[self.ConfigurationMetaNames.DESCRIPTION] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

            if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                metadata[self.ConfigurationMetaNames.SPACE_UID] = {
                    "href": "/v4/spaces/" + meta_props[self.ConfigurationMetaNames.SPACE_UID]
                }
            if self._client.ICP_30:
                if self.ConfigurationMetaNames.COMMAND in meta_props:
                    metadata[self.ConfigurationMetaNames.COMMAND] = meta_props[self.ConfigurationMetaNames.COMMAND]
                if self.ConfigurationMetaNames.CUSTOM in meta_props:
                    metadata[self.ConfigurationMetaNames.CUSTOM] = meta_props[self.ConfigurationMetaNames.CUSTOM]

            # Following is not used for model_definitions since space_id/project_id
            # is passed as a query param. Code is existing pre-cloud-convergence
            if self._client.CAMS:
                if self._client.default_space_id is not None:
                    metadata['space'] = {'href': "/v4/spaces/"+self._client.default_space_id}
                elif self._client.default_project_id is not None:
                    metadata['project'] = {'href': "/v2/projects/"+self._client.default_project_id}
                else:
                    raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        document = self._generate_model_definition_document(meta_props)

        if self._client.WSD:
            return self._wsd_create_asset("wml_model_definition", document, meta_props, model_definition,user_archive_file=True)

        model_definition_attachment_def = {
          "asset_type": "wml_model_definition",
          "name": "model_definition_attachment"
        }
        paramvalue = self._client._params()

        try:
            if not self._ICP:
                creation_response = requests.post(
                     self._href_definitions.get_model_definition_assets_href(),
                     params=paramvalue,
                     headers=self._client._get_headers(),
                     json=document)
            else:
                creation_response = requests.post(
                    self._href_definitions.get_model_definition_assets_href(),
                    params=paramvalue,
                    headers=self._client._get_headers(),
                    json=document,
                    verify=False)

            model_definition_details = self._handle_response(201, u'creating new model_definition', creation_response)

            if self._client.CLOUD_PLATFORM_SPACES:
                model_definition_attachment_url = self._href_definitions.get_attachments_href(model_definition_details['metadata']['asset_id'])
            else:
                model_definition_attachment_url = self._href_definitions.get_model_definition_assets_href() + "/" + \
                                                  model_definition_details['metadata']['asset_id'] + "/attachments"

            put_header = self._client._get_headers(no_content_type=True)
            files = {'file': open(model_definition, 'rb')}
            if creation_response.status_code == 201:
                model_definition_id = model_definition_details['metadata']['asset_id']
                if not self._ICP:
                    attachment_response = requests.post(
                        model_definition_attachment_url,
                        params=paramvalue,
                        headers=self._client._get_headers(),
                        json=model_definition_attachment_def,
                        verify=False)
                else:
                    attachment_response = requests.post(
                         model_definition_attachment_url,
                         params=paramvalue,
                         headers=self._client._get_headers(),
                         json=model_definition_attachment_def,
                         verify=False)
                attachment_details = self._handle_response(201, u'creating new attachment', attachment_response)
                if attachment_response.status_code == 201:
                    attachment_id = attachment_details['attachment_id']
                    attachment_status_json = json.loads(attachment_response.content.decode("utf-8"))
                    model_definition_attachment_signed_url = attachment_status_json["url1"]
                 #   print("WML model_definition attachment url1: %s" % model_definition_attachement_signed_url)
                    model_definition_attachment_put_url = self._client.wml_credentials['url'] + model_definition_attachment_signed_url

                    if not self._ICP:
                        # On Cloud, additional line gets added to the attachment
                        # multi form data. Use data instead
                        put_response = requests.put(model_definition_attachment_signed_url,
                                                    data=open(model_definition, 'rb').read())
                                                    # # files=files,
                                                    # params=paramvalue,
                                                    # headers=put_header)
                    else:

                        put_response = requests.put(model_definition_attachment_put_url,
                                                    files=files,
                                                    verify=False)
                    if put_response.status_code == 201 or put_response.status_code == 200:
                        complete_url = self._href_definitions.get_attachment_href(model_definition_id, attachment_id) \
                                       + "/complete"
                        if not self._ICP:
                            complete_response = requests.post( self._href_definitions.get_attachment_complete_href(model_definition_id, attachment_id),
                                                              params=paramvalue,
                                                              headers=self._client._get_headers(),
                                                              verify=False)
                        else:
                            complete_response = requests.post(complete_url,
                                                              params=paramvalue,
                                                              headers=self._client._get_headers(),
                                                              verify=False)

                        if complete_response.status_code == 200:
                            response = self._get_required_element_from_response(model_definition_details)

                            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                                return response
                            else:
                                entity = response[u'entity']

                                try:
                                    del entity[u'wml_model_definition'][u'ml_version']
                                except KeyError:
                                    pass

                                final_response = {
                                    "metadata": response[u'metadata'],
                                    "entity": entity
                                }

                                return final_response
                            # return self._get_required_element_from_response(model_definition_details)
                        else:
                            self._delete(model_definition_id)
                            raise WMLClientError("Failed while creating a model_definition. Try again.")
                    else:
                        self._delete(model_definition_id)
                        raise WMLClientError("Failed while creating a model_definition. Try again.")
                else:
                    self._delete(model_definition_id)
                    raise WMLClientError("Failed while creating a model_definition. Try again.")
            else:
                raise WMLClientError("Failed while creating a model_definition. Try again.")

        except Exception as e:

            raise e


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, model_definition_uid):
        """
           Get metadata of stored model_definition.


           **Parameters**

           .. important::
                #. **model_definition_uid**: Unique Id of model_definition \n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: metadata of model definition\n
                **return type**: dict (if model_definition_uid is not None)\n

           **Example**

            >>> model_definition_details = client.model_definitions.get_details(model_definition_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        op_name = 'getting model_definition details'
        modeldef_uid = str_type_conv(model_definition_uid)
        ModelDefinition._validate_type(modeldef_uid, u'model_definition_uid', STR_TYPE, False)

        url = self._href_definitions.get_model_definition_assets_href() + u'/' + modeldef_uid
        paramvalue = self._client._params()
        if not self._ICP:
            response_get = requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers()
            )
        else:
            response_get = requests.get(
                url,
                params=paramvalue,
                headers=self._client._get_headers(),
                verify=False
            )
        if response_get.status_code == 200:
            get_model_definition_details = self._handle_response(200, op_name, response_get)
            response = self._get_required_element_from_response(get_model_definition_details)

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:

                entity = response[u'entity']

                try:
                    del entity[u'wml_model_definition'][u'ml_version']
                except KeyError:
                    pass

                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": entity
                }

                return final_response
            # return self._get_required_element_from_response(get_model_definition_details)
        else:
            return self._handle_response(200, op_name, response_get)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def download(self, model_definition_uid, filename, rev_id=None):
        """
            Download the content of a script asset.

            **Parameters**

            .. important::
                 #. **model_definition_uid**:  The Unique Id of the model_definition asset to be downloaded\n
                    **type**: str\n

                 #. **filename**:  filename to be used for the downloaded file\n
                    **type**: str\n

                 #. **rev_id**:  Revision id\n
                    **type**: str\n

            **Output**

            .. important::

                 **returns**: Path to the downloaded asset content\n
                 **return type**: str\n

            **Example**

             >>> client.script.download(asset_uid,"script_file.zip")
        """

        self._client._check_if_either_is_set()

        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', STR_TYPE, True)
        params = self._client._params()
        if rev_id is not None:
            ModelDefinition._validate_type(rev_id, u'rev_id', int, False)
            params.update({'revision_id': rev_id})

        if self._client.WSD:
            import urllib
            response = requests.get(self._href_definitions.get_data_asset_href(model_definition_uid),
                                              params=self._client._params(),
                                              headers=self._client._get_headers())

            model_def_details = self._handle_response(200, u'get model', response)
            attachment_url = model_def_details['attachments'][0]['object_key']
            attachment_signed_url = self._href_definitions.get_wsd_model_attachment_href() + \
                                   urllib.parse.quote('wml_model_definition/' + attachment_url, safe='')

        else:
            attachment_id = self._get_attachment_id(model_definition_uid)
            artifact_content_url = self._href_definitions.get_attachment_href(model_definition_uid, attachment_id)
            if not self._ICP and not self._WSD:
                response = requests.get(self._href_definitions.get_attachment_href(model_definition_uid, attachment_id), params=self._client._params(),
                                    headers=self._client._get_headers())
            else:
                response = requests.get(artifact_content_url, params=self._client._params(),
                                    headers=self._client._get_headers(), verify=False)
            attachment_signed_url = response.json()["url"]
        if response.status_code == 200:
            if not self._ICP and not self._client.WSD:
                if self._client.CLOUD_PLATFORM_SPACES:
                    att_response = requests.get(attachment_signed_url)
                else:
                    att_response = requests.get(self._wml_credentials["url"] + attachment_signed_url)
            else:
                if self._client.WSD:
                    att_response = requests.get(attachment_signed_url, params= self._client._params(),
                                                headers=self._client._get_headers(),
                                                stream=True, verify=False)
                else:
                    att_response = requests.get(self._wml_credentials["url"]+attachment_signed_url,

                                                verify=False)
            if att_response.status_code != 200:
                raise WMLClientError(u'Failure during {}.'.format("downloading model_definition asset"),
                                     att_response)

            downloaded_asset = att_response.content
            try:
                with open(filename, 'wb') as f:
                    f.write(downloaded_asset)
                print(u'Successfully saved asset content to file: \'{}\''.format(filename))
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(u'Saving asset with artifact_url: \'{}\' failed.'.format(filename), e)
        else:
            raise WMLClientError("Failed while downloading the asset " + model_definition_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, model_definition_uid):
        """
            Delete a stored model_definition.

            **Parameters**

            .. important::
                #. **model_definition_uid**: Unique Id of stored Model definition\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.model_definitions.delete(model_definition_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        model_definition_uid = str_type_conv(model_definition_uid)
        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', STR_TYPE, True)
        paramvalue = self._client._params()

        model_definition_endpoint = self._href_definitions.get_model_definition_assets_href() + "/" + model_definition_uid
        if not self._ICP:
            response_delete = requests.delete(model_definition_endpoint, params=paramvalue, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(model_definition_endpoint, params=paramvalue, headers=self._client._get_headers(), verify=False)

        return self._handle_response(204, u'Model definition deletion', response_delete, False)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _delete(self, model_definition_uid):
        """
            Delete a stored model_definition.

            **Parameters**

            .. important::
                #. **model_definition_uid**: Unique Id of Model definition\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.model_definitions.delete(model_definition_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        model_definition_uid = str_type_conv(model_definition_uid)
        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', STR_TYPE, True)
        paramvalue = self._client._params()
        model_definition_endpoint = self._href_definitions.get_model_definition_assets_href() + "/" + model_definition_uid
        if not self._ICP:
            response_delete = requests.delete(model_definition_endpoint, params=paramvalue, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(model_definition_endpoint, params=paramvalue, headers=self._client._get_headers(), verify=False)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data,  u'model_definition_response', dict)
        revision_id = None

        try:
            href = ""
            if self._client.default_space_id is not None:
                new_el = {'metadata': {'space_id': response_data['metadata']['space_id'],
                                   'guid': response_data['metadata']['asset_id'],
                                   'asset_type': response_data['metadata']['asset_type'],
                                   'created_at': response_data['metadata']['created_at'],
                                   'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                                },
                      'entity': response_data['entity']

                      }
                href = "/v2/assets/" + response_data['metadata']['asset_type'] + "/" + response_data['metadata'][
                    'asset_id'] + "?" + "space_id=" + response_data['metadata']['space_id']

            elif self._client.default_project_id is not None:
                new_el = {'metadata': {'project_id': response_data['metadata']['project_id'],
                                       'guid': response_data['metadata']['asset_id'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at'],
                                       'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                                       },
                          'entity': response_data['entity']

                          }

                href = "/v2/assets/" + response_data['metadata']['asset_type'] + "/" + response_data['metadata'][
                    'asset_id'] + "?" + "project_id=" + response_data['metadata']['project_id']

            if 'revision_id' in response_data['metadata']:
                new_el['metadata'].update({'revision_id': response_data['metadata']['revision_id']})
                revision_id = response_data[u'metadata'][u'revision_id']

            if 'name' in response_data['metadata']:
                new_el['metadata'].update({'name': response_data['metadata']['name']})

            if 'description' in response_data['metadata'] and response_data['metadata']['description']:
                new_el['metadata'].update({'description': response_data['metadata']['description']})

            if 'href' in response_data['metadata']:
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    href_without_host = response_data['href'].split('.com')[-1]
                    new_el[u'metadata'].update({'href': href_without_host})
                else:
                    new_el['metadata'].update({'href': response_data['href']})
            else:
                new_el['metadata'].update({'href': href})

            if "attachments" in response_data and response_data[u'attachments']:
                new_el['metadata'].update({'attachment_id': response_data[u'attachments'][0][u'id']})
            else:
                new_el['metadata'].update({'href': href})

            if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                new_el['metadata'].update(
                    {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data)

    def _get_attachment_id(self, model_definition_uid, rev_id=None):
        op_name = 'getting attachment id '
        url = self._href_definitions.get_model_definition_assets_href() + u'/' + model_definition_uid
        paramvalue = self._client._params()

        if rev_id is not None:
            paramvalue.update({'revision_id': rev_id})

        if not self._ICP:
            response_get = requests.get(
                url,
                params=paramvalue,
                headers=self._client._get_headers()
            )
        else:
            response_get = requests.get(
                url,
                params=paramvalue,
                headers=self._client._get_headers(),
                verify=False
            )
        details = self._handle_response(200, op_name, response_get)
        attachment_id = details["attachments"][0]["id"]
        return attachment_id

    def list(self, limit=None):
        """
           List stored model_definition assets. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all model_definition assets in a table format.\n
                **return type**: None\n

           **Example**

                     >>> client.model_definitions.list()

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        href = self._href_definitions.get_model_definition_search_asset_href()
        if limit is None:
            data = {
                "query": "*:*"
            }
        else:
            ModelDefinition._validate_type(limit, u'limit', int, False)
            data = {
                "query": "*:*",
                "limit": limit
            }

        if not self._ICP:
            response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),json=data)
        else:
            response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),json=data, verify=False)
        self._handle_response(200, u'model_definition assets', response)
        asset_details = self._handle_response(200, u'model_definition assets', response)["results"]
        model_def_values = [
            (m[u'metadata'][u'name'], m[u'metadata'][u'asset_type'], m[u'metadata'][u'asset_id']) for
            m in asset_details]

        self._list(model_def_values, [u'NAME', u'ASSET_TYPE', u'GUID'], limit, _DEFAULT_LIST_LENGTH)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(self, model_definition_details):
        """
                Get Unique Id of stored model_definition asset.

                **Parameters**

                .. important::

                   #. **model_definition_details**:  Metadata of the stored model_definition asset\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of stored model_definition asset\n
                    **return type**: str\n

                **Example**

                     >>> asset_uid = client.model_definition.get_id(asset_details)

        """

        return ModelDefinition.get_uid(self, model_definition_details)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(self, model_definition_details):
        """
        Get uid of stored model. Deprecated!! Use get_id(model_definition_details) instead

        **Parameters**

        .. important::

            #. **param model_definition_details**:  stored model_definition details\n
               **type**: dict

        **Output**

        .. important::

            **returns**: uid of stored model_definition\n
            **return type**: str\n

        **Example**

            >>> model_definition_uid = client.model_definitions.get_uid(model_definition_details)
        """
        if 'asset_id' in model_definition_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details',
                                                               [u'metadata', u'asset_id'])
        else:
            ModelDefinition._validate_type(model_definition_details, u'model__definition_details', object, True)
            #ModelDefinition._validate_type_of_details(model_definition_details, MODEL_DEFINITION_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details', [u'metadata', u'guid'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(self, model_definition_details):
        """
        Get href of stored model_definition.

        **Parameters**

        .. important::
            #. **model_definition_details**: Stored model_definition details.\n
               **type**: dict\n

        **Output**

        .. important::
            **returns**: href of stored model_definition.\n
            **return type**: str\n

        **Example**
            >>> model_definition_uid = client.model_definitions.get_href(model_definition_details)
        """
        if 'asset_id' in model_definition_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details', [u'metadata', u'asset_id'])
        else:
            ModelDefinition._validate_type(model_definition_details, u'model__definition_details', object, True)
            # ModelDefinition._validate_type_of_details(model_definition_details, MODEL_DEFINITION_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details',
                                                               [u'metadata', u'href'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update(self, model_definition_id, meta_props=None, file_path=None):
        """
        Update model_definition with either metadata or attachment or both.\n

        **Parameters**

        .. important::

            #. **model_definition_id**:  model_definition ID.\n
               **type**: str\n

            #. **file_path**: Path to the content file to be uploaded.\n
               **type**: str\n

        ** Output**

        .. important::

            **returns**: Updated metadata of model_definition.\n
            **return type**: dict\n

        **Example**

            >>> model_definition_details = client.model_definition.update(model_definition_id, meta_props, file_path)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        # We need to enable this once we add functionality for WSD
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ModelDefinition._validate_type(model_definition_id, 'model_definition_id', STR_TYPE, True)

        if meta_props is None and file_path is None:
            raise WMLClientError('Atleast either meta_props or file_path has to be provided')

        updated_details = None

        url = self._href_definitions.get_asset_href(model_definition_id)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        #   b. Construct entity patch if required and call /v2/assets/<asset_id>/attributes/script to patch entity
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the model_definition, delete it
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
                    u'Invalid input. Unable to get the details of model_definition_id provided.')
            else:
                raise WMLClientError(u'Failure during {}.'.format("getting script to update"), response)

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

            props_for_asset_entity_patch = {}
            for key in meta_props:
                if key != 'name' and key != 'description':
                    props_for_asset_entity_patch.update({key: meta_props[key]})

            entity_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity']['wml_model_definition'],
                                                                                       props_for_asset_entity_patch,
                                                                                       with_validation=True,
                                                                                       asset_meta_patch=False)

            if meta_patch_payload:
                meta_patch_url = self._href_definitions.get_asset_href(model_definition_id)

                if not self._ICP:
                    response_patch = requests.patch(meta_patch_url,
                                                    json=meta_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers())
                else:
                    response_patch = requests.patch(meta_patch_url,
                                                    json=meta_patch_payload,
                                                    params=self._client._params(),
                                                    headers=self._client._get_headers(),
                                                    verify=False)

                updated_details = self._handle_response(200, u'script patch', response_patch)

            if entity_patch_payload:
                entity_patch_url = self._href_definitions.get_asset_href(model_definition_id) + \
                                   '/attributes/wml_model_definition'

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
            attachments_response = self._update_attachment_for_assets("wml_model_definition",
                                                                      model_definition_id,
                                                                      file_path,
                                                                      current_attachment_id)

        if attachments_response is not None and 'success' not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._href_definitions.get_asset_href(model_definition_id)

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
                    u'Invalid input. Unable to get the details of model_definition_id provided.')
            else:
                raise WMLClientError(u'Failure during {}.'.format("getting script to update"), response)

        response = self._get_required_element_from_response(self._handle_response(200, "Get script details", response))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'wml_model_definition'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

        # return self._get_required_element_from_response(response)

    def _update_msg(self, updated_details):
        if updated_details is not None:
            print("Could not update the attachment because of server error."
                  " However metadata is updated. Try updating attachment again later")
        else:
            raise WMLClientError('Unable to update attachment because of server error. Try again later')

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_revision(self, model_definition_uid):
        """
        Creates revision for the given model_definition. Revisions are immutable once created.
        The metadata and attachment at model_definition is taken and a revision is created out of it.

        **Parameters**

        .. important::

            #. **model_definition**: model_definition ID.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Stored model_definition revisions metadata.\n
            **return type**: dict\n

        **Example**

            >>> model_definition_revision = client.model_definitions.create_revision(model_definition_id)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')

        self._client._check_if_either_is_set()

        model_defn_id = str_type_conv(model_definition_uid)
        ModelDefinition._validate_type(model_defn_id, u'model_defn_id', STR_TYPE, True)

        print("Creating model_definition revision...")

        # return self._get_required_element_from_response(
        #     self._create_revision_artifact_for_assets(model_defn_id, 'Model definition'))

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(model_defn_id, 'Model definition'))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'wml_model_definition'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_revision_details(self, model_definition_uid, rev_uid=None):
        """
        Get metadata of model_definit

        **Parameters**

        .. important::

            #. **model_definition_uid**: model_definition ID.\n
               **type**: str\n

            #. **rev_uid**: Revision ID. If this parameter is not provided, returns latest revision if existing else error.\n
               **type**: int\n

        **Output**

        .. important::

            **returns**: Stored model definitions metadata.\n
            **return type**: dict\n

        **Example**

            >>> script_details = client.model_definitions.get_revision_details(model_definition_uid, rev_uid)
        """

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')
        op_name = 'getting model_definition revision details'
        modeldef_uid = str_type_conv(model_definition_uid)
        ModelDefinition._validate_type(modeldef_uid, u'model_definition_uid', STR_TYPE, True)
        ModelDefinition._validate_type(modeldef_uid, u'model_definition_uid', STR_TYPE, True)

        url = self._href_definitions.get_model_definition_assets_href() + u'/' + modeldef_uid
        paramvalue = self._client._params()

        if rev_uid is None:
            rev_uid = 'latest'

        paramvalue.update({'revision_id': rev_uid})

        if not self._ICP:
            response_get = requests.get(
                url,
                params=paramvalue,
                headers=self._client._get_headers()
            )
        else:
            response_get = requests.get(
                url,
                params=paramvalue,
                headers=self._client._get_headers(),
                verify=False
            )
        if response_get.status_code == 200:
            # get_model_definition_details = self._handle_response(200, op_name, response_get)
            # return self._get_required_element_from_response(get_model_definition_details)

            response = self._get_required_element_from_response(self._handle_response(200, op_name, response_get))

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                entity = response[u'entity']

                try:
                    del entity[u'wml_model_definition'][u'ml_version']
                except KeyError:
                    pass

                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": entity
                }

                return final_response
        else:
            return self._handle_response(200, op_name, response_get)


    def list_revisions(self, model_definition_uid, limit=None):
        """
           List stored model_definition assets. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **model_definition_uid**:  Unique id of model_definition\n
                   **type**: str\n

                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all model_definition revision in a table format.\n
                **return type**: None\n

           **Example**

                >>> client.model_definitions.list_revisions()

        """
        ##For CP4D, check if either spce or project ID is set
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')
        self._client._check_if_either_is_set()
        href = self._href_definitions.get_model_definition_assets_href() + "/" + model_definition_uid +\
               u'/revisions'
        params = self._client._params()
        #params = None
        if limit is not None:
            ModelDefinition._validate_type(limit, u'limit', int, False)
            params.update( {
                "limit": limit
            })
        if not self._ICP:
            response = requests.get(href, params, headers=self._client._get_headers())
        else:
            response = requests.get(href, params=params, headers=self._client._get_headers(), verify=False)
        self._handle_response(200, u'model_definition revision assets', response)
        asset_details = self._handle_response(200, u'model_definition revision assets', response)["results"]
        model_def_values = [
            (m[u'metadata'][u'asset_id'],
             m[u'metadata'][u'revision_id'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'asset_type'],
             m[u'metadata'][u'commit_info'][u'committed_at']) for
            m in asset_details]

        self._list(model_def_values, [u'GUID', u'REV_ID', u'NAME', u'ASSET_TYPE', u'REVISION_COMMIT'],
                   limit,
                   _DEFAULT_LIST_LENGTH)



