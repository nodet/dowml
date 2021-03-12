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
from ibm_watson_machine_learning.utils import SPACES_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, MEMBER_DETAILS_TYPE,PKG_EXTN_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv, str_type_conv, get_file_from_cos
from ibm_watson_machine_learning.metanames import PkgExtnMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os,json

_DEFAULT_LIST_LENGTH = 50


class PkgExtn(WMLResource):
    """
    Store and manage your software Packages Extension specs.

    """
    ConfigurationMetaNames = PkgExtnMetaNames()
    """MetaNames for Package Extensions creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, pkg_extn_uid):
        """
            Get package extensions details.

            **Parameters**

            .. important::
                #. **pkg_extn_details**:  Metadata of the package extensions\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: pkg_extn UID\n
                **return type**: str

            **Example**

             >>> pkg_extn_details = client.pkg_extn.get_details(pkg_extn_uid)
        """
        PkgExtn._validate_type(pkg_extn_uid, u'pkg_extn_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.get(self._href_definitions.get_pkg_extn_href(pkg_extn_uid), params=self._client._params(),
                                    headers=self._client._get_headers())
        else:
            response = requests.get(self._href_definitions.get_pkg_extn_href(pkg_extn_uid), params=self._client._params(),
                                      headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            return self._get_required_element_from_response(self._handle_response(200, u'get hw spec details', response))
        else:
            return self._handle_response(200, u'get hw spec details', response)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props, file_path):
        """
                Create a package extensions.

                **Parameters**

                .. important::
                   #. **meta_props**:  meta data of the pacakge extension. To see available meta names use:\n
                                    >>> client.package_extensions.ConfigurationMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of the package extensions\n
                    **return type**: dict\n

                **Example**

                 >>> meta_props = {
                 >>>    client.package_extensions.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
                 >>>    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
                 >>>    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml",
                 >>> }

                **Example**

                 >>> pkg_extn_details = client.package_extensions.store(meta_props=meta_props,file_path="/path/to/file")

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        PkgExtn._validate_type(meta_props, u'meta_props', dict, True)
        pkg_extn_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        pkg_extn_meta_json = json.dumps(pkg_extn_meta)

        PkgExtn._validate_type(file_path, u'file_path', str, True)
        ##Step1: Create a pkg extn
        f = {'file': open(file_path, 'rb')}
        #Step1  : Create an asset
        print("Creating package extensions")
        href = self._href_definitions.get_pkg_extns_href()

        if not self._ICP:
            creation_response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(), data=pkg_extn_meta_json)
        else:
            creation_response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(), data=pkg_extn_meta_json, verify=False)

        pkg_extn_details = self._handle_response(201, u'creating new package_extensions', creation_response)

        # Step2: upload pkg extension file to presigned url
        if creation_response.status_code == 201:
            pkg_extn_asset_id = pkg_extn_details["metadata"]["asset_id"]
            pkg_extn_presigned_url = pkg_extn_details["entity"]["package_extension"]["href"]

            if not self._ICP:
                put_response = requests.put(
                    pkg_extn_presigned_url,
                    data=open(file_path, 'rb').read(),
                )
            else:
                put_response = requests.put(
                    self._wml_credentials['url'] + pkg_extn_presigned_url,
                    files=f,
                    verify=False
                )
            if put_response.status_code == 201 or put_response.status_code == 200:
                # Step3: Mark the upload complete
                if not self._ICP:
                    complete_response = requests.post(
                        self._href_definitions.get_pkg_extn_href(pkg_extn_asset_id) + "/upload_complete",
                        headers=self._client._get_headers(),
                        params=self._client._params()
                    )
                else:
                    complete_response = requests.post(
                        self._href_definitions.get_pkg_extn_href(pkg_extn_asset_id) + "/upload_complete",
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                        verify=False
                    )
                if complete_response.status_code == 204:
                    print("SUCCESS")
                    return self._get_required_element_from_response(pkg_extn_details)
                else:
                    #print(complete_response.text) # remove print later
                    self._delete(pkg_extn_asset_id)
                    raise WMLClientError("Failed while creating a package extensions " + complete_response.text)
            else:
                self._delete(pkg_extn_asset_id)
                raise WMLClientError("Failed while creating a package extensions " + put_response.text)
        else:
            raise WMLClientError("Failed while creating a package extensions " + creation_response.text)

    def list(self):
        """
           List package extensions.

           **Output**

           .. important::
                This method only prints the list of all package extensionss in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.package_extensions.list()

        """

        href = self._href_definitions.get_pkg_extns_href()

        if not self._ICP:
            response = requests.get(href, params=self._client._params(), headers=self._client._get_headers())
        else:
            response = requests.get(href, params=self._client._params(), headers=self._client._get_headers(), verify=False)
        self._handle_response(200, u'list pkg_extn', response)
        asset_details = self._handle_response(200, u'list assets', response)["resources"]
        pkg_extn_values = [
            (m[u'metadata'][u'name'],
             m[u'metadata'][u'asset_id'],
             m[u'entity'][u'package_extension'][u'type'],
             m[u'metadata'][u'created_at']) for
            m in asset_details]

        self._list(pkg_extn_values, [u'NAME', u'ASSET_ID', u'TYPE', u'CREATED_AT'], None, _DEFAULT_LIST_LENGTH)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(pkg_extn_details):
        """
                Get Unique Id of package extensions. Deprecated!! use get_id(pkg_extn_details) instead

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the package extensions \n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of package extension \n
                    **return type**: str\n

                **Example**


                 >>> asset_uid = client.package_extensions.get_uid(pkg_extn_details)

        """
        PkgExtn._validate_type(pkg_extn_details, u'pkg_extn_details', object, True)
        #print(pkg_extn_details)
        PkgExtn._validate_type_of_details(pkg_extn_details, PKG_EXTN_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(pkg_extn_details, u'pkg_extn_details',
                                                           [u'metadata', u'asset_id'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(pkg_extn_details):
        """
                Get Unique Id of package extensions.

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the package extensions \n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of package extension \n
                    **return type**: str\n

                **Example**


                 >>> asset_id = client.package_extensions.get_id(pkg_extn_details)

        """

        return PkgExtn.get_uid(pkg_extn_details)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid_by_name(self, pkg_extn_name):
        """
                Get UID of package extensions. Deprecated!! Use get_id_by_name(pkg_extn_name) instead

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the package extension\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of package extension \n
                    **return type**: str\n

                **Example**


                 >>> asset_uid = client.package_extensions.get_uid_by_name(pkg_extn_name)
        """
        PkgExtn._validate_type(pkg_extn_name, u'pkg_extn_name', STR_TYPE, True)


        parameters = self._client._params()
        parameters.update(name=pkg_extn_name)
        if not self._ICP:
            response = requests.get(self._href_definitions.get_pkg_extns_href(),
                                    params=parameters,
                                    headers=self._client._get_headers())
        else:

            response = requests.get(self._href_definitions.get_pkg_extns_href(),
                                    params=parameters,
                                    headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            total_values = self._handle_response(200, u'get pkg extn', response)["total_results"]
            if total_values != 0:
                pkg_extn_details = self._handle_response(200, u'get pkg extn', response)["resources"]
                return pkg_extn_details[0][u'metadata'][u'asset_id']
            else:
                return "Not Found"


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id_by_name(self, pkg_extn_name):
        """
                Get ID of package extensions.

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the package extension\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of package extension \n
                    **return type**: str\n

                **Example**


                 >>> asset_id = client.package_extensions.get_id_by_name(pkg_extn_name)
        """
        return PkgExtn.get_uid_by_name(self, pkg_extn_name)


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(pkg_extn_details):
        """
            Get url of stored package extensions.

           **Parameters**

           .. important::
                #. **asset_details**:  package extensions details\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of package extensions details\n
                **return type**: str\n

           **Example**

             >>> pkg_extn_details = client.package_extensions.get_details(pkg_extn_uid)
             >>> pkg_extn_href = client.package_extensions.get_href(pkg_extn_details)

        """
        PkgExtn._validate_type(pkg_extn_details, u'pkg_extn_details', object, True)
        PkgExtn._validate_type_of_details(pkg_extn_details, PKG_EXTN_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(pkg_extn_details,
                                                           u'pkg_extn_details',
                                                           [u'entity', u'package_extension', u'href'])


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, pkg_extn_uid):
        """
            Delete a package extension.

            **Parameters**

            .. important::
                #. **pkg_extn_uid**: Unique Id of package extension\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.package_extensions.delete(pkg_extn_uid)

        """
        PkgExtn._validate_type(pkg_extn_uid, u'pkg_extn_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_pkg_extn_href(pkg_extn_uid), params=self._client._params(),
                                    headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_pkg_extn_href(pkg_extn_uid), params=self._client._params(),
                                      headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete pkg extn specification', response)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _delete(self, pkg_extn_uid):
        PkgExtn._validate_type(pkg_extn_uid, u'pkg_extn_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_pkg_extn_href(pkg_extn_uid), params=self._client._params(),
                                       headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_pkg_extn_href(pkg_extn_uid), params=self._client._params(),
                                       headers=self._client._get_headers(), verify=False)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'pkg_extn_response', dict)
        try:
            if self._client.default_space_id is not None:
                new_el = {'metadata': {'space_id': response_data['metadata']['space_id'],
                                       'name': response_data['metadata']['name'],
                                       'asset_id': response_data['metadata']['asset_id'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at']
                                       #'updated_at': response_data['metadata']['updated_at']
                                       },
                          'entity': response_data['entity']

                          }
            elif self._client.default_project_id is not None:
                if self._client.WSD:

                    href = "/v2/assets/" + response_data['metadata']['asset_id'] + "?" + "project_id=" + response_data['metadata']['project_id']

                    new_el = {'metadata': {'project_id': response_data['metadata']['project_id'],
                                           'name': response_data['metadata']['name'],
                                           'asset_id': response_data['metadata']['asset_id'],
                                           'asset_type': response_data['metadata']['asset_type'],
                                           'created_at': response_data['metadata']['created_at']
                                           },
                              'entity': response_data['entity']

                              }
                else:
                    new_el = {'metadata': {'project_id': response_data['metadata']['project_id'],
                                           'name': response_data['metadata']['name'],
                                           'asset_id': response_data['metadata']['asset_id'],
                                           'asset_type': response_data['metadata']['asset_type'],
                                           'created_at': response_data['metadata']['created_at']
                                       },
                             'entity': response_data['entity']

                            }
            if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'href' in response_data['metadata']:
                href_without_host = response_data['metadata']['href'].split('.com')[-1]
                new_el[u'metadata'].update({'href': href_without_host})

            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data.text)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def download(self, pkg_extn_id, filename):
        """
            Download a package extension.

            **Parameters**

            .. important::
                 #. **pkg_extn_uid**:  The Unique Id of the package extension to be downloaded\n
                    **type**: str\n

                 #. **filename**:  filename to be used for the downloaded file\n
                    **type**: str\n

            **Output**

            .. important::
                 **returns**: Path to the downloaded package extension content\n
                 **return type**: str\n

            **Example**

             >>> client.assets.download(asset_uid,"sample_conda.yml/custom_library.zip")

         """

        PkgExtn._validate_type(pkg_extn_id, u'pkg_extn_id', STR_TYPE, True)

        if self._WSD:
            import urllib
            pkg_extn_response = requests.get(self._href_definitions.get_pkg_extn_href(pkg_extn_id),
                                          params=self._client._params(),
                                          headers=self._client._get_headers())

            pkg_extn_details = self._handle_response(200, u'get assets', pkg_extn_response)

            artifact_content_url = pkg_extn_details['entity']['package_extension']['href']

            r = requests.get(artifact_content_url, params=self._client._params(), headers=self._client._get_headers(),
                             stream=True, verify=False)
            if r.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), r)

            downloaded_asset = r.content
            try:
                with open(filename, 'wb') as f:
                    f.write(downloaded_asset)
                print(u'Successfully saved asset content to file: \'{}\''.format(filename))
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(u'Saving asset with artifact_url: \'{}\' failed.'.format(filename), e)
        else:
            if not self._ICP:
                pkg_extn_response = requests.get(self._href_definitions.get_pkg_extn_href(pkg_extn_id),
                                          params=self._client._params(),
                                          headers=self._client._get_headers())
            else:
                pkg_extn_response = requests.get(self._href_definitions.get_pkg_extn_href(pkg_extn_id),
                                                 params=self._client._params(),
                                                 headers=self._client._get_headers(),
                                                 verify=False)

            pkg_extn_details = self._handle_response(200, u'get assets', pkg_extn_response)

            artifact_content_url = pkg_extn_details['entity']['package_extension']['href']

            if pkg_extn_response.status_code == 200:
                if not self._ICP:
                    # att_response = requests.get(self._wml_credentials["url"]+artifact_content_url)
                    att_response = requests.get(artifact_content_url)
                else:
                    att_response = requests.get(self._wml_credentials["url"]+artifact_content_url,
                                            verify=False)

                if att_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading package extension"),
                                            att_response)

                downloaded_asset = att_response.content
                try:
                    with open(filename, 'wb') as f:
                        f.write(downloaded_asset)
                    print(u'Successfully saved package extension content to file: \'{}\''.format(filename))
                    return os.getcwd() + "/" + filename
                except IOError as e:
                    raise WMLClientError(u'Saving asset with artifact_url: \'{}\' failed.'.format(filename), e)
            else:
                raise WMLClientError("Failed while downloading the package extension "+ pkg_extn_id)