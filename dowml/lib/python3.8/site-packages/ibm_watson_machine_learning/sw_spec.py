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
from ibm_watson_machine_learning.utils import SPACES_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, MEMBER_DETAILS_TYPE,SW_SPEC_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv, str_type_conv, get_file_from_cos
from ibm_watson_machine_learning.metanames import SwSpecMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os,json

_DEFAULT_LIST_LENGTH = 50


class SwSpec(WMLResource):
    """
    Store and manage your software specs.

    """
    ConfigurationMetaNames = SwSpecMetaNames()
    """MetaNames for Software Specification creation."""
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP
        self.software_spec_list = None
        if self._client.WSD_20:
            self.software_spec_list = {
                                "default_py3.6": "0062b8c9-8b7d-44a0-a9b9-46c416adcbd9",
                                "scikit-learn_0.20-py3.6": "09c5a1d0-9c1e-4473-a344-eb7b665ff687",
                                "ai-function_0.1-py3.6": "0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda",
                                "shiny-r3.6": "0e6e79df-875e-4f24-8ae9-62dcc2148306",
                                "pytorch_1.1-py3.6": "10ac12d6-6b30-4ccd-8392-3e922c096a92" ,
                                "scikit-learn_0.22-py3.6": "154010fa-5b3b-4ac1-82af-4d5ee5abbc85",
                                "default_r3.6":  "1b70aec3-ab34-4b87-8aa0-a4a3c8296a36",
                                "tensorflow_1.15-py3.6":  "2b73a275-7cbf-420b-a912-eae7f436e0bc",
                                "pytorch_1.2-py3.6":  "2c8ef57d-2687-4b7d-acce-01f94976dac1",
                                "spark-mllib_2.3":  "2e51f700-bca0-4b0d-88dc-5c6791338875",
                                "pytorch-onnx_1.1-py3.6-edt": "32983cea-3f32-4400-8965-dde874a8d67e",
                                "spark-mllib_2.4": "390d21f8-e58b-4fac-9c55-d7ceda621326",
                                "xgboost_0.82-py3.6":  "39e31acd-5f30-41dc-ae44-60233c80306e",
                                "pytorch-onnx_1.2-py3.6-edt": "40589d0e-7019-4e28-8daa-fb03b6f4fe12",
                                "spark-mllib_2.4-r_3.6": "49403dff-92e9-4c87-a3d7-a42d0021c095",
                                "xgboost_0.90-py3.6": "4ff8d6c2-1343-4c18-85e1-689c965304d3",
                                "pytorch-onnx_1.1-py3.6":  "50f95b2a-bc16-43bb-bc94-b0bed208c60b",
                                "spark-mllib_2.4-scala_2.11": "55a70f99-7320-4be5-9fb9-9edb5a443af5",
                                "spss-modeler_18.1":  "5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b",
                                "spark-mllib_2.3-r_3.6":  "6586b9e3-ccd6-4f92-900f-0f8cb2bd6f0c",
                                "spss-modeler_18.2":  "687eddc9-028a-4117-b9dd-e57b36f1efa5",
                                "pytorch-onnx_1.2-py3.6":  "692a6a4d-2c4d-45ff-a1ed-b167ee55469a",
                                "do_12.9":  "75a3a4b0-6aa0-41b3-a618-48b1f56332a6",
                                "spark-mllib_2.3-scala_2.11": "7963efe5-bbec-417e-92cf-0574e21b4e8d",
                                "caffe_1.0-py3.6":  "7bb3dbe2-da6e-4145-918d-b6d84aa93b6b",
                                "cuda-py3.6":  "82c79ece-4d12-40e6-8787-a7b9e0f62770",
                                "hybrid_0.1":  "8c1a58c6-62b5-4dc4-987a-df751c2756b6",
                                "caffe-ibm_1.0-py3.6":  "8d863266-7927-4d1e-97d7-56a7f4c0a19b",
                                "spss-modeler_17.1":  "902d0051-84bd-4af6-ab6b-8f6aa6fdeabb",
                                "do_12.10":  "9100fd72-8159-4eb9-8a0b-a87e12eefa36",
                                "hybrid_0.2":  "9b3f9040-9cee-4ead-8d7a-780600f542f7"}

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self,sw_spec_uid=None):
        """
            Get software specification details.

            **Parameters**

            .. important::
                #. **sw_spec_details**:  Metadata of the stored sw_spec\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: sw_spec UID\n
                **return type**: str

            **Example**

             >>> sw_spec_details = client.software_specifications.get_details(sw_spec_uid)

        """
        if self._client.WSD_20:
            raise WMLClientError(u'get_details API is not supported in Watson Studio Desktop.')

        SwSpec._validate_type(sw_spec_uid, u'sw_spec_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.get(self._href_definitions.get_sw_spec_href(sw_spec_uid),
                                    params=self._client._params(skip_space_project_chk=True),
                                    headers=self._client._get_headers())
        else:
            if self._client.ICP_PLATFORM_SPACES:
                response = requests.get(self._href_definitions.get_sw_spec_href(sw_spec_uid),
                                        params=self._client._params(skip_space_project_chk=True),
                                        headers=self._client._get_headers(), verify=False)
            else:
                response = requests.get(self._href_definitions.get_sw_spec_href(sw_spec_uid), params=self._client._params(),
                                          headers=self._client._get_headers(), verify=False)

        if response.status_code == 200:
            return self._get_required_element_from_response(self._handle_response(200, u'get sw spec details', response))
        else:
            return self._handle_response(200, u'get sw spec details', response)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props):
        """
                Create a software specification.

                **Parameters**

                .. important::
                   #. **meta_props**:  meta data of the space configuration. To see available meta names use:\n
                                    >>> client.software_specifications.ConfigurationMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of the stored space\n
                    **return type**: dict\n

                **Example**

                 >>> meta_props = {
                 >>>    client.software_specifications.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
                 >>>    client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
                 >>>    client.software_specifications.ConfigurationMetaNames.PACKAGE_EXTENSIONS_UID: [],
                 >>>    client.software_specifications.ConfigurationMetaNames.SOFTWARE_CONFIGURATIONS: {},
                 >>>    client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION_ID: "guid"
                 >>> }

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.WSD_20:
            raise WMLClientError(u'store() API is not supported in Watson Studio Desktop.')

        SwSpec._validate_type(meta_props, u'meta_props', dict, True)
        sw_spec_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        sw_spec_meta_json = json.dumps(sw_spec_meta)
        href = self._href_definitions.get_sw_specs_href()

        if not self._ICP:
            creation_response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(), data=sw_spec_meta_json)
        else:
            creation_response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(), data=sw_spec_meta_json, verify=False)

        sw_spec_details = self._handle_response(201, u'creating sofware specifications', creation_response)

        return sw_spec_details


    def list(self):
        """
           List software specifications.

           **Output**

           .. important::
                This method only prints the list of all software specifications in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.software_specifications.list()
        """

        if not self._client.WSD_20:
            href = self._href_definitions.get_sw_specs_href()
            if not self._ICP:
                response = requests.get(href, params=self._client._params(), headers=self._client._get_headers())
            else:
                response = requests.get(href, params=self._client._params(), headers=self._client._get_headers(), verify=False)
            self._handle_response(200, u'list sw_specs', response)
            asset_details = self._handle_response(200, u'list assets', response)["resources"]
            sw_spec_values = [
                (m[u'metadata'][u'name'], m[u'metadata'][u'asset_id'],m[u'entity'][u'software_specification'].get('type', 'derived')) for
                m in asset_details]

            self._list(sw_spec_values, [u'NAME', u'ASSET_ID', u'TYPE'], None, _DEFAULT_LIST_LENGTH)
        else:
            from tabulate import tabulate
            header = [u'NAME', u'ASSET_ID', u'TYPE']
            print(tabulate(self.software_spec_list.items(), headers=header))


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(sw_spec_details):
        """
                Get Unique Id of software specification.

                **Parameters**

                .. important::

                   #. **sw_spec_details**:  Metadata of the software specification\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of software specification \n
                    **return type**: str\n

                **Example**


                 >>> asset_uid = client.software_specifications.get_id(sw_spec_details)

        """

        return SwSpec.get_uid(sw_spec_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(sw_spec_details):
        """
                Get Unique Id of software specification. Deprecated!! Use get_id(sw_spec_details) instead

                **Parameters**

                .. important::

                   #. **sw_spec_details**:  Metadata of the software specification\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of software specification \n
                    **return type**: str\n

                **Example**


                 >>> asset_uid = client.software_specifications.get_uid(sw_spec_details)

        """
        SwSpec._validate_type(sw_spec_details, u'sw_spec_details', object, True)
        SwSpec._validate_type_of_details(sw_spec_details, SW_SPEC_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(sw_spec_details, u'sw_spec_details',
                                                           [u'metadata', u'asset_id'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id_by_name(self, sw_spec_name):
        """
                Get Unique Id of software specification.

                **Parameters**

                .. important::

                   #. **sw_spec_name**:  Name of the software specification\n
                      **type**: str\n

                **Output**

                .. important::

                    **returns**: Unique Id of software specification \n
                    **return type**: str\n

                **Example**


                 >>> asset_uid = client.software_specifications.get_id_by_name(sw_spec_name)

        """

        return SwSpec.get_uid_by_name(self, sw_spec_name)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid_by_name(self, sw_spec_name):
        """
                Get Unique Id of software specification. Deprecated!! Use get_id_by_name(self, sw_spec_name) instead

                **Parameters**

                .. important::

                   #. **sw_spec_name**:  Name of the software specification\n
                      **type**: str\n

                **Output**

                .. important::

                    **returns**: Unique Id of software specification \n
                    **return type**: str\n

                **Example**


                 >>> asset_uid = client.software_specifications.get_uid_by_name(sw_spec_name)

        """

        SwSpec._validate_type(sw_spec_name, u'sw_spec_uid', STR_TYPE, True)
        if not self._client.WSD_20:
            parameters = self._client._params(skip_space_project_chk=True)
            parameters.update(name=sw_spec_name)
            if not self._ICP:
                response = requests.get(self._href_definitions.get_sw_specs_href(),
                                        params=parameters,
                                        headers=self._client._get_headers())
            else:

                response = requests.get(self._href_definitions.get_sw_specs_href(),
                                        params=parameters,
                                        headers=self._client._get_headers(), verify=False)
            if response.status_code == 200:
                total_values = self._handle_response(200, u'list assets', response)["total_results"]
                if total_values != 0:
                    sw_spec_details = self._handle_response(200, u'list assets', response)["resources"]
                    return sw_spec_details[0][u'metadata'][u'asset_id']
                else:
                    return "Not Found"
        else:
            return self.software_spec_list.get(sw_spec_name)


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(sw_spec_details):
        """
            Get url of software specification.

           **Parameters**

           .. important::
                #. **sw_spec_details**:  software specification details\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of software specification \n
                **return type**: str\n

           **Example**

             >>> sw_spec_details = client.software_specifications.get_details(sw_spec_uid)
             >>> sw_spec_href = client.software_specifications.get_href(sw_spec_details)

        """
        SwSpec._validate_type(sw_spec_details, u'sw_spec_details', object, True)
        SwSpec._validate_type_of_details(sw_spec_details, SW_SPEC_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(sw_spec_details, u'sw_spec_details', [u'metadata', u'href'])


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, sw_spec_uid):
        """
            Delete a software specification.

            **Parameters**

            .. important::
                #. **sw_spec_uid**: Unique Id of software specification\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.software_specifications.delete(sw_spec_uid)

        """
        if self._client.WSD_20:
            raise WMLClientError(u'delete API is not supported in Watson Studio Desktop.')

        SwSpec._validate_type(sw_spec_uid, u'sw_spec_uid', STR_TYPE, True)

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_sw_spec_href(sw_spec_uid), params=self._client._params(),
                                    headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_sw_spec_href(sw_spec_uid), params=self._client._params(),
                                      headers=self._client._get_headers(), verify=False)
        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete software specification', response)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def add_package_extension(self, sw_spec_uid, pkg_extn_id):
        """
                Add a package extension to software specifications existing metadata.

                **Parameters**

                .. important::

                    #. **sw_spec_uid**:  Unique Id of software specification which should be updated\n
                       **type**: str\n
                    #. **pkg_extn_id**:  Unique Id of package extension which should needs to added to software specification\n
                       **type**: str\n

                **Example**

                >>> client.software_specifications.add_package_extension(sw_spec_uid, pkg_extn_id)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.WSD_20:
            raise WMLClientError(u'package extension APIs are not supported in Watson Studio Desktop.')

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        sw_spec_uid = str_type_conv(sw_spec_uid)
        self._validate_type(sw_spec_uid, u'sw_spec_uid', STR_TYPE, True)
        self._validate_type(pkg_extn_id, u'pkg_extn_id', STR_TYPE, True)

        url = self._href_definitions.get_sw_spec_href(sw_spec_uid)

        url = url + "/package_extensions/" + pkg_extn_id


        if not self._ICP:
            response = requests.put(url, params=self._client._params(), headers=self._client._get_headers())
        else:
            response = requests.put(url,  params=self._client._params(), headers=self._client._get_headers(), verify=False)

        if response.status_code == 204:
            print("SUCCESS")
            return "SUCCESS"
        else:
            return self._handle_response(204, u'pkg spec add', response, False)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete_package_extension(self, sw_spec_uid, pkg_extn_id):
        """
                Delete a package extension from software specifications existing metadata.

                **Parameters**

                .. important::

                    #. **sw_spec_uid**:  Unique Id of software specification which should be updated\n
                       **type**: str\n
                    #. **pkg_extn_id**:  Unique Id of package extension which should needs to deleted from software specification\n
                       **type**: str\n

                **Example**

                 >>> client.software_specifications.delete_package_extension(sw_spec_uid, pkg_extn_id)

        """
        if self._client.WSD_20:
            raise WMLClientError(u'package extension APIs are not supported in Watson Studio Desktop.')

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        sw_spec_uid = str_type_conv(sw_spec_uid)
        self._validate_type(sw_spec_uid, u'sw_spec_uid', STR_TYPE, True)
        self._validate_type(pkg_extn_id, u'pkg_extn_id', STR_TYPE, True)

        url = self._href_definitions.get_sw_spec_href(sw_spec_uid)

        url = url + "/package_extensions/" + pkg_extn_id

        if not self._ICP:
            response = requests.delete(url,
                                       params=self._client._params(),
                                       headers=self._client._get_headers())
        else:
            response = requests.delete(url,
                                       params=self._client._params(),
                                       headers=self._client._get_headers(),
                                       verify=False)

        return self._handle_response(204, u'pkg spec delete', response, False)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'sw_spec_response', dict)
        try:
            if self._client.default_space_id is not None:
                new_el = {'metadata': {
                                       'name': response_data['metadata']['name'],
                                       'asset_id': response_data['metadata']['asset_id'],
                                       'href': response_data['metadata']['href'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at']
                                       #'updated_at': response_data['metadata']['updated_at']
                                       },
                          'entity': response_data['entity']

                          }
            elif self._client.default_project_id is not None:
                if self._client.WSD:

                    href = "/v2/assets/" + response_data['metadata']['asset_id'] + "?" + "project_id=" + response_data['metadata']['project_id']

                    new_el = {'metadata': {
                                           'name': response_data['metadata']['name'],
                                           'asset_id': response_data['metadata']['asset_id'],
                                           'href': response_data['metadata']['href'],
                                           'asset_type': response_data['metadata']['asset_type'],
                                           'created_at': response_data['metadata']['created_at']
                                           },
                              'entity': response_data['entity']

                              }
                else:
                    new_el = {'metadata': {
                                           'name': response_data['metadata']['name'],
                                           'asset_id': response_data['metadata']['asset_id'],
                                           'href': response_data['metadata']['href'],
                                           'asset_type': response_data['metadata']['asset_type'],
                                           'created_at': response_data['metadata']['created_at']
                                       },
                             'entity': response_data['entity']

                            }

            else:
                # For system software spec
                new_el = {'metadata': {
                                       'name': response_data['metadata']['name'],
                                       'asset_id': response_data['metadata']['asset_id'],
                                       'href': response_data['metadata']['href'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at']
                                       #'updated_at': response_data['metadata']['updated_at']
                                       },
                          'entity': response_data['entity']

                          }
            if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'href' in response_data['metadata']:
                href_without_host = response_data['metadata']['href'].split('.com')[-1]
                new_el[u'metadata'].update({'href': href_without_host})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data.text)
