from __future__ import print_function
import requests
from ibm_watson_machine_learning.utils import STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv
from ibm_watson_machine_learning.metanames import ExportMetaNames
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
_DEFAULT_LIST_LENGTH = 50

class Export(WMLResource):
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._client = client
        self.ConfigurationMetaNames = ExportMetaNames()

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def start(self, meta_props, space_id=None, project_id=None):
        """
            Start the export. Either space_id or project_id has to be provided and is mandatory.
            ALL_ASSETS is by default False. No need to provide explicitly unless it has to be set to True
            Either ALL_ASSETS or ASSET_TYPES or ASSET_IDS has to be given in the meta_props. Only one of these can be
            provided

            In the meta_props:

            ALL_ASSETS is a boolean. When set to True, it exports all assets in the given space
            ASSET_IDS is an array containing the list of assets ids to be exported
            ASSET_TYPES is for providing the asset types to be exported. All assets of that asset type will be exported
                        Eg: wml_model, wml_model_definition, wml_pipeline, wml_function, wml_experiment,
                        software_specification, hardware_specification, package_extension, script

           **Parameters**

           .. important::
                #. **meta_props**:  meta data. To see available meta names use **client.export_assets.ConfigurationMetaNames.get()**\n
                   **type**: dict\n
                #. **space_id**:  Space identifier**\n
                   **type**: str\n
                #. **project_id**:  Project identifier**\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: Response json\n
                **return type**: dict\n

           **Example**

                >>> metadata = {
                >>>    client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                >>>    client.export_assets.ConfigurationMetaNames.ASSET_IDS: ["13a53931-a8c0-4c2f-8319-c793155e7517", "13a53931-a8c0-4c2f-8319-c793155e7518"]
                >>> }
                >>> details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

                >>> metadata = {
                >>>    client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                >>>    client.export_assets.ConfigurationMetaNames.ASSET_TYPES: ["wml_model"],
                >>> }
                >>> details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

                >>> metadata = {
                >>>    client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                >>>    client.export_assets.ConfigurationMetaNames.ALL_ASSETS: True,
                >>> }
            """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Either 'space_id' or 'project_id' has to be provided")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        Export._validate_type(meta_props, u'meta_props', dict, True)
        self._validate_input_meta(meta_props)

        meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        start_meta = {}
        assets = {}

        start_meta[u'name'] = meta[u'name']
        if 'description' in meta:
            start_meta[u'description'] = meta[u'description']

        if "all_assets" not in meta:
            assets.update({"all_assets": False})
        else:
            assets.update({"all_assets": meta[u'all_assets']})

        if "asset_types" in meta:
            assets.update({"asset_types": meta[u'asset_types']})

        if "asset_ids" in meta:
            assets.update(({"asset_ids": meta[u'asset_ids']}))

        start_meta[u'assets'] = assets

        href = self._href_definitions.exports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        if not self._ICP:
            creation_response = requests.post(href,
                                              params=params,
                                              headers=self._client._get_headers(),
                                              json=start_meta)
        else:
            creation_response = requests.post(href,
                                              params=params,
                                              headers=self._client._get_headers(),
                                              json=start_meta,
                                              verify=False)

        details = self._handle_response(expected_status_code=202,
                                        operationName=u'export start',
                                        response=creation_response)

        export_id = details[u'metadata']['id']

        print("export job with id {} has started. Monitor status using client.export_assets.get_details api. "
              "Check 'help(client.export_assets.get_details)' for details on the api usage".format(export_id))

        return details

    def _validate_input_meta(self, meta_props):
        if 'name' not in meta_props:
            raise WMLClientError("Its mandatory to provide 'NAME' in meta_props. Example: "
                                 "client.export_assets.ConfigurationMetaNames.NAME: 'name'")

        if 'all_assets' not in meta_props and 'asset_ids' not in meta_props and 'asset_types' not in meta_props:
            raise WMLClientError("Its mandatory to provide either 'ALL_ASSETS' or 'ASSET_IDS' or 'ASSET_TYPES' " 
                                 "in meta_props. Example: client.export_assets.ConfigurationMetaNames.ALL_ASSETS: True")

        count = 0

        if 'all_assets' in meta_props:
            count = count + 1
        if 'asset_ids' in meta_props:
            count = count + 1
        if 'asset_types' in meta_props:
            count = count + 1

        if count > 1:
            raise WMLClientError("Only one of 'ALL_ASSETS' or 'ASSET_IDS' or 'ASSET_TYPES' can be provided")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def cancel(self, export_id, space_id=None, project_id=None):
        """
            Cancel an export job. 'export_id' and 'space_id'( or 'project_id' has to be provided )
            Note: To delete a export_id job, use delete() api

            **Parameters**

            .. important::
                #. **export_id**:  Export job identifier\n
                   **type**: str\n
                #. **space_id**:  Space identifier\n
                   **type**: str\n
                #. **project_id**:  Project identifier\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.export_assets.cancel(export_id='6213cf1-252f-424b-b52d-5cdd9814956c',
             >>>                      space_id='3421cf1-252f-424b-b52d-5cdd981495fe')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        Export._validate_type(export_id, u'export_id', STR_TYPE, True)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        href = self._href_definitions.export_href(export_id)

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        if not self._ICP:
            cancel_response = requests.delete(href,
                                              params=params,
                                              headers=self._client._get_headers())
        else:
            cancel_response = requests.delete(href,
                                              params=params,
                                              headers=self._client._get_headers(),
                                              verify=False)

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'cancel export',
                                        response=cancel_response)

        if "SUCCESS" == details:
            print("Export job cancelled")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, export_id, space_id=None, project_id=None):
        """
            Deletes the given export_id job. 'space_id' or 'project_id' has to be provided

            **Parameters**

            .. important::
                #. **export_id**:  Export job identifier\n
                   **type**: str\n
                #. **space_id**:  Space identifier**\n
                   **type**: str\n
                #. **project_id**:  Project identifier**\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.export_assets.delete(export_id='6213cf1-252f-424b-b52d-5cdd9814956c',
             >>>                      space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        Export._validate_type(export_id, u'export_id', STR_TYPE, True)

        href = self._href_definitions.export_href(export_id)

        params = {"hard_delete": True}

        if space_id is not None:
            params.update({'space_id': space_id})
        else:
            params.update({'project_id': project_id})

        if not self._ICP:
            delete_response = requests.delete(href,
                                              params=params,
                                              headers=self._client._get_headers())
        else:
            delete_response = requests.delete(href,
                                              params=params,
                                              headers=self._client._get_headers(),
                                              verify=False)

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'delete export job',
                                        response=delete_response)

        if "SUCCESS" == details:
            print("Export job deleted")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, export_id=None, space_id=None, project_id=None, limit=None):
        """
            Get metadata of the given export job. if no export_id is specified all
            exports metadata is returned.

           **Parameters**

           .. important::
                #. **export_id**: export job identifier (optional)\n
                   **type**: str\n
                #. **space_id**:  Space identifier**\n
                   **type**: str\n
                #. **project_id**:  Project identifier**\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: export(s) metadata\n
                **return type**: dict\n
                dict (if export_id is not None) or {"resources": [dict]} (if export_id is None)\n

           .. note::
                If export_id is not specified, all export(s) metadata is fetched\n

           **Example**

             >>> details = client.export_assets.get_details(export_id, space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
             >>> details = client.export_assets.get_details()
         """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")
        
        Export._validate_type(export_id, u'export_id', STR_TYPE, False)
        Export._validate_type(limit, u'limit', int, False)

        href = self._href_definitions.exports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        return self._get_artifact_details(href, export_id, limit, 'export job', query_params=params)
    
    def list(self, space_id=None, project_id=None, limit=None):
        """
            List export jobs. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n
                #. **space_id**:  Space identifier**\n
                   **type**: str\n
                #. **project_id**:  Project identifier**\n
                   **type**: str\n

           **Output**

           .. important::
                This method only prints the list of all export jobs in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.export_assets.list()
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        if space_id is not None:
            resources = self.get_details(space_id=space_id)[u'resources']
        else:
            resources = self.get_details(project_id=project_id)[u'resources']


        values = [(m[u'metadata'][u'id'],
                   m[u'metadata'][u'name'],
                   m[u'metadata'][u'created_at'],
                   m[u'entity'][u'status'][u'state']) for m in resources]

        self._list(values, [u'ID', u'NAME', u'CREATED', u'STATUS'], limit, _DEFAULT_LIST_LENGTH)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(export_details):
        """
                Get ID of export job from export details

                **Parameters**

                .. important::

                   #. **export_details**:  Metadata of the export job\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: ID of the export job\n
                    **return type**: str\n

                **Example**

                 >>> id = client.export_assets.get_id(export_details)
        """
        Export._validate_type(export_details, u'export_details', object, True)

        return WMLResource._get_required_element_from_dict(export_details,
                                                           u'export_details',
                                                           [u'metadata', u'id'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_exported_content(self, export_id, space_id=None, project_id=None, file_path=None):
        """
            Get the exported content as a zip file

            **Parameters**

            .. important::
                #. **export_id**:  export job identifier\n
                   **type**: str\n
                #. **space_id**:  Space identifier**\n
                   **type**: str\n
                #. **project_id**:  Project identifier**\n
                   **type**: str\n
                #. **file_path**:  name of local file to create. This should be absolute path of the file and
                                  the file shouldn't exist**\n
                   **type**: str\n


            **Output**

            .. important::

               **returns**: Path to the downloaded function content\n
               **return type**: str\n

            **Example**

             >>> client.exports.get_exported_content(export_id,
             >>>                                     space_id= '98a53931-a8c0-4c2f-8319-c793155e4598'
             >>>                                     file_path='/home/user/my_exported_content.zip')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        if os.path.isfile(file_path):
            raise WMLClientError(u'File with name: \'{}\' already exists.'.format(file_path))

        Export._validate_type(file_path, u'file_path', STR_TYPE, True)

        href = self._href_definitions.export_content_href(export_id)

        try:
            if not self._ICP:
                response = requests.get(href,
                                        params=params,
                                        headers=self._client._get_headers(),
                                        stream=True)
            else:
                response = requests.get(href,
                                        params=params,
                                        headers=self._client._get_headers(),
                                        stream=True,
                                        verify=False)

            if response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading export content"), response)

            downloaded_exported_content = response.content
            self._logger.info(u'Successfully downloaded artifact with artifact_url: {}'.format(href))
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(u'Downloading export content with artifact_url: \'{}\' failed.'.format(href), e)

        try:
            with open(file_path, 'wb') as f:
                f.write(downloaded_exported_content)
            print(u'Successfully saved export content to file: \'{}\''.format(file_path))
            return file_path
        except IOError as e:
            raise WMLClientError(u'Downloading export content with artifact_url: \'{}\' failed.'.format(href), e)

