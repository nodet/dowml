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
from ibm_watson_machine_learning.metanames import VolumeMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os
import shlex
import subprocess
_DEFAULT_LIST_LENGTH = 50


class Volume(WMLResource):
    """
    Store and manage your scripts assets.

    """
    ConfigurationMetaNames = VolumeMetaNames()
    """MetaNames for script Assets creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, volume_id):
        """
            Get Volume  details.

            **Parameters**

            .. important::
                #. **volume_name**: Unique name  of the volume\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: Metadata of the volume details \n
                **return type**: dict\n

            **Example**

             >>> volume_details = client.volumes.get_details(volume_name)

        """
        Volume._validate_type(volume_id, u'volume_id', STR_TYPE, True)

        params = {'addon_type': 'volumes',
                  'include_service_status':True
                  }
        if not self._ICP:
            response = requests.get(self._href_definitions.volume_href(volume_id),
                                    headers=self._client._get_headers(zen=True))
        else:
            response = requests.get(self._href_definitions.volume_href(volume_id),
                                      headers=self._client._get_headers(zen=True), verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code, response.text)
            raise WMLClientError("Failed to Get the volume details. Try again.")


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create(self, meta_props):
        """
                Creates a Volume asset.

                **Parameters**

                .. important::
                   #. **meta_props**:  Name to be given to the Volume asset\n

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of the created volume details\n
                    **return type**: dict\n

                **Example**
                Provision new PVC volume :
                 >>> metadata = {
                 >>>        client.volumes.ConfigurationMetaNamess.NAME: 'volume-for-wml-test',
                 >>>        client.volumes.ConfigurationMetaNames.NAMESPACE: 'wmldev2',
                 >>>        client.volumes.ConfigurationMetaNames.STORAGE_CLASS: 'nfs-client'
                 >>>        client.volumes.ConfigurationMetaNames.STORAGE_SIZE: "2G"
                 >>>    }
                 >>>
                 >>> asset_details = client.scripts.store(meta_props=metadata)

               Provision a Existing PVC volume:

                 >>> metadata = {
                 >>>        client.volumes.ConfigurationMetaNamess.NAME: 'volume-for-wml-test',
                 >>>        client.volumes.ConfigurationMetaNames.NAMESPACE: 'wmldev2',
                 >>>        client.volumes.ConfigurationMetaNames.EXISTING_PVC_NAME: 'volume-for-wml-test'
                 >>>    }
                 >>>
                 >>> asset_details = client.scripts.store(meta_props=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Failed to create volume. It is supported only for CP4D 3.5")

        volume_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        create_meta = {}
        if self.ConfigurationMetaNames.EXISTING_PVC_NAME in meta_props and \
            meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME] is not None:
            if self.ConfigurationMetaNames.STORAGE_CLASS in meta_props and \
                    meta_props[self.ConfigurationMetaNames.STORAGE_CLASS] is not None:
                raise WMLClientError("Failed while creating volume. Either provide EXISTING_PVC_NAME to create a volume using existing volume or"
                                     "provide STORAGE_CLASS and STORAGE_SIZE for new volume creation")
            else:
                create_meta.update({ "existing_pvc_name": meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME]})
        else:
            if self.ConfigurationMetaNames.STORAGE_CLASS in meta_props and \
               meta_props[self.ConfigurationMetaNames.STORAGE_CLASS] is not None:
               if self.ConfigurationMetaNames.STORAGE_SIZE in meta_props and \
                       meta_props[self.ConfigurationMetaNames.STORAGE_SIZE] is not None:
                   create_meta.update({"storageClass": meta_props[self.ConfigurationMetaNames.STORAGE_CLASS]})
                   create_meta.update({"storageSize": meta_props[self.ConfigurationMetaNames.STORAGE_SIZE]})
               else:
                   raise WMLClientError("Failed to create volume. Missing input STORAGE_SIZE" )

        if self.ConfigurationMetaNames.EXISTING_PVC_NAME in meta_props and meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME] is not None:
            input_meta = {
                "addon_type":"volumes",
                "addon_version":"-",
                "create_arguments":{
                    "metadata":create_meta
                },
                "namespace":meta_props[self.ConfigurationMetaNames.NAMESPACE],
                "display_name":meta_props[self.ConfigurationMetaNames.NAME]
            }
        else:
            input_meta = {
                "addon_type": "volumes",
                "addon_version": "-",
                "create_arguments": {
                    "metadata": create_meta
                },
                "namespace": meta_props[self.ConfigurationMetaNames.NAMESPACE],
                "display_name": meta_props[self.ConfigurationMetaNames.NAME]
            }
        creation_response = {}
        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                creation_response = requests.post(
                    self._href_definitions.volumes_href(),
                    headers=self._client._get_headers(zen=True),
                    json=input_meta
                )

            else:
                creation_response = requests.post(self._href_definitions.volumes_href(),
                        headers=self._client._get_headers(zen=True),
                        json=input_meta,
                        verify=False
                    )
            if creation_response.status_code == 200:
                volume_id_details = creation_response.json()
                import copy
                volume_details = copy.deepcopy(input_meta)
                volume_details.update(volume_id_details)
                return volume_details
            else:
                print(creation_response.status_code, creation_response.text)
                raise WMLClientError("Failed to create a volume. Try again.")
        except Exception as e:
            print("Exception: ", {e})
            raise WMLClientError("Failed to create a volume. Try again.")



    def start(self, name):
        """
            Start the  volume service.

            **Parameters**

            .. important::
                #. **volume_name**:  Unique name of the volume to be started
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

                >>> client.volumes.start(volume_name)

        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.ICP_PLATFORM_SPACES and not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("Volume APIs are not supported. It is supported only for CP4D 3.5")
        start_url = self._href_definitions.volume_service_href(name)
        # Start the volume  service
        start_data = {}
        try:
            if not self._ICP:
                start_data = {}
                creation_response = requests.post(
                    start_url,
                    headers=self._client._get_headers(zen=True),
                    json=start_data
                )
            else:
                creation_response = requests.post(
                   start_url,
                    headers=self._client._get_headers(zen=True),
                    json=start_data,
                    verify=False
                )
            if creation_response.status_code == 200:
                print("Volume Service started")
            elif creation_response.status_code == 500:
                print("Failed to start the volume. Make sure volume is in running with status RUNNING or UNKNOW and then re-try")
            else:
                print(creation_response.status_code, creation_response.text)
                raise WMLClientError("Failed to start the file to  volume. Try again.")
        except Exception as e:
            print("Exception:", {e})
            raise WMLClientError("Failed to start the file to  volume. Try again.")

    def upload_file(self, name,  file_path):
        """
            Upload the data file into stored volume.

            **Parameters**

                .. important::

                #. **name**:  Unique name of the stored volume \n
                   **type**: str\n

                #. **file_path**:  File to be uploaded into the volume \n
                   **type**: str\n

            **Output**
                .. important::

                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

                **Example**

                    >>> client.volumes.upload_file('testA', 'DRUG.csv')

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.ICP_PLATFORM_SPACES and not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("Volume APIs are not supported. It is supported only for CP4D 3.5")

        header_input = self._client._get_headers(zen=True)
        zen_token = header_input.get('Authorization')

        filename_to_upload = file_path.split('/')[-1]
        upload_url_file = self._href_definitions.volume_upload_href(name) + filename_to_upload
        cmd_str = 'curl -k  -X PUT "' + upload_url_file + '"' + "  -H 'Content-Type: multipart/form-data' -H 'Authorization: " + zen_token + \
                  "' -F upFile='@" + file_path + "'"
        args = shlex.split(cmd_str)
        upload_response = subprocess.run(args, capture_output=True, text=True)
        if upload_response.returncode == 0:
            import json
            try:
                cmd_output = json.loads(upload_response.stdout)
                print(cmd_output.get('message'))
                return "SUCCESS"
            except Exception as e:
                print(upload_response.returncode, upload_response.stdout)
                print("Failed to upload the file to  volume. Try again.")
                return "FAILED"
        else:
            print(upload_response.returncode, upload_response.stdout, upload_response.stderr)
            print("Failed to upload the file to  volume. Try again.")
            return "FAILED"

       # header = {'Authorization': zentoken}
       #  dirfiles = {'file': None}
       #  if not self._ICP:
       #      upload_dir_response = requests.post(
       #          upload_url,
       #          headers=header,
       #          files=dirfiles
       #          )
       #  else:
       #      upload_dir_response = requests.put(
       #          upload_url,
       #          headers=header,
       #          files=dirfiles,
       #          verify=False
       #      )
       #  if upload_dir_response.status_code != 200:
       #      raise WMLClientError("Failed to create a directory to upload the file to  volume. Try again.")
       #
       #
       #  filename_to_upload = file_path.split('/')[-1]
       #  #upload_url_file = self._href_definitions.volume_upload_href(name) +
       #  upload_url_file = self._href_definitions.volume_upload_href(name) + "wmldir/"
       # #  #upload_url_file = self._href_definitions.volume_upload_href(name) +
       # #  import urllib
       # #  upload_url_file = self._href_definitions.volume_upload_href(name)  + urllib.parse.quote(filename_to_upload, safe='')
       # #  print(upload_url_file)
       #  import urllib
       #  upload_url_file = self._href_definitions.volume_upload_href(name) + filename_to_upload
       #  #upload_url_file = urllib.parse.quote(self._href_definitions.volume_upload_href(name) + "wmldir/" +filename_to_upload, safe='')
       #  files = {'upload_file': (filename_to_upload,  open(file_path, 'rb'))}
       #
       #  with open(file_path, 'rb') as f:
       #      if not self._ICP:
       #          upload_response = requests.put(
       #              upload_url_file,
       #              headers=header,
       #              files=files
       #              )
       #      else:
       #          f = {'upload_file': (filename_to_upload, open(file_path, 'rb'), 'csv', {'Expires': '0'})}
       #          # with open(file_path, 'rb') as f:
       #          #     fdata = f.read()
       #              # upload_response = requests.put(
       #              #     upload_url_file,
       #              #     files={'file': (filename_to_upload, fdata, 'text/csv', {'Expires': '0'})},
       #              #     #files=files,
       #              #     #files=f,
       #              #     headers=header,
       #              #     verify=False
       #              # )
       #          # header.update({'Accept': 'text/csv'})#',image/png,image/gif',
       #          # upload_response = requests.request("PUT", upload_url_file,
       #          #                  files=f,
       #          #                  headers=header, verify=False)

    def list(self):
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

            >>> client.volumes.list()
        """

        href = self._href_definitions.volumes_href()
        params = {}
        params.update({'addon_type': 'volumes'})
        if not self._ICP:
            response = requests.get(href, params=params, headers=self._client._get_headers(zen=True))
        else:
            response = requests.get(href, params=params, headers=self._client._get_headers(zen=True), verify=False)
        asset_details = self._handle_response(200, u'list volumes', response)
        asset_list = asset_details.get('service_instances')
        volume_values = [
            (m[u'display_name'],
             m[u'id'],
             m['provision_status']) for m in asset_list]

        self._list(volume_values, [u'NAME', u'ID', u'PROVISION_STATUS'], None, _DEFAULT_LIST_LENGTH)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(volume_details):
        """
                Get Unique Id of stored volume details.

                **Parameters**

                .. important::

                   #. **asset_details**:  Metadata of the stored volume details\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique Id of stored volume asset\n
                    **return type**: str\n

                **Example**

                     >>> volume_uid = client.volumes.get_id(volume_details)

        """

        Volume._validate_type(volume_details, u'asset_details', object, True)
        if 'service_instance' in volume_details and  volume_details.get('service_instance') is not None:
            vol_details = volume_details.get('service_instance')
            return WMLResource._get_required_element_from_dict(vol_details, u'volume_assets_details',
                                                               [u'id'])
        else:
            return WMLResource._get_required_element_from_dict(volume_details, u'volume_assets_details',
                                                           [u'id'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_name(volume_details):
        """
                Get Unique name  of stored volume asset.

                **Parameters**

                .. important::

                   #. **volume_details**:  Metadata of the stored volume asset\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Unique name of stored volume asset\n
                    **return type**: str\n

                **Example**

                 >>> volume_name = client.volumes.get_name(asset_details)

        """
        Volume._validate_type(volume_details, u'asset_details', object, True)
        if 'service_instance' in volume_details and  volume_details.get('service_instance') is not None:
            vol_details = volume_details.get('service_instance')
            return WMLResource._get_required_element_from_dict(vol_details, u'volume_assets_details',
                                                               [u'display_name'])
        else:
            return WMLResource._get_required_element_from_dict(volume_details, u'volume_assets_details',
                                                           [u'display_name'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, volume_id):
        """
            Delete a volume.

            **Parameters**

            .. important::
                #. **volume_name**:  Unique name of the volume
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.volumes.delete(volume_name)

        """
        Volume._validate_type(volume_id, u'asset_uid', STR_TYPE, True)
        if (not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES):
            raise WMLClientError(u'Volume API is not supported.')

        if not self._ICP:
            response = requests.delete(self._href_definitions.volume_href(volume_id),
                                       headers=self._client._get_headers(zen=True))
        else:
            response = requests.delete(self._href_definitions.volume_href(volume_id),
                                      headers=self._client._get_headers(zen=True), verify=False)

        if response.status_code == 200 or response.status_code == 204:
            print("Successfully deleted volume service.")
            return "SUCCESS"
        else:
            print("Failed to delete volume.")
            print(response.status_code, response.text)
            return "FAILED"

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def stop(self, volume_name):
        """
            Stop the  volume service.

            **Parameters**

            .. important::
                #. **volume_name**:  Unique name of the volume
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.volumes.stop(volume_name)

        """
        Volume._validate_type(volume_name, u'asset_uid', STR_TYPE, True)
        if (not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES):
            raise WMLClientError(u'Volume API is not supported.')

        if not self._ICP:
            response = requests.delete(self._href_definitions.volume_service_href(volume_name),
                                       headers=self._client._get_headers(zen=True))
        else:
            response = requests.delete(self._href_definitions.volume_service_href(volume_name),
                                       headers=self._client._get_headers(zen=True), verify=False)
        if response.status_code == 200:
            print("Successfully stopped volume service.")
            return "SUCCESS"
        else:
            print(response.status_code, response.text)
            return "FAILED"

