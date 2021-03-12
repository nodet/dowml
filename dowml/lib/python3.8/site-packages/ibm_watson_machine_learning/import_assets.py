from __future__ import print_function
import requests
from ibm_watson_machine_learning.utils import STR_TYPE, STR_TYPE_NAME, docstring_parameter, meta_props_str_conv
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
_DEFAULT_LIST_LENGTH = 50

class Import(WMLResource):
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._client = client

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def start(self, file_path=None, space_id=None, project_id=None):
        """
            Start the import. Either space_id or project_id has to be provided and is mandatory. Note that
            on IBM Cloud Pak® for Data for Data 3.5, import into non-empty space/project is not supported

           **Parameters**

           .. important::
                #. **meta_props**:  meta data. To see available meta names use **client.import_assets.ConfigurationMetaNames.get()**\n
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

                >>> details = client.import_assets.start(space_id="98a53931-a8c0-4c2f-8319-c793155e4598",
                >>>                                      file_path="/home/user/data_to_be_imported.zip")
            """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Either 'space_id' or 'project_id' has to be provided")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        if file_path is None:
            raise WMLClientError("Its mandatory to provide 'file_path'")

        if not os.path.isfile(file_path):
            raise WMLClientError(u'File with name: \'{}\' does not exist'.format(file_path))

        # with open(file_path, "rb") as a_file:
        #     file_dict = {file_path: a_file}

        # file_handler = {'file.zip': open(file_path,'rb')}

        file = open(file_path, 'rb')

        # from datetime import datetime
        # start_time = datetime.now()
        #
        data = file.read()
        #
        # end_time = datetime.now()
        #
        # dt = end_time - start_time
        # ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0

        # print("Time taken to read the zip file in ms: {}".format(ms))

        href = self._href_definitions.imports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        from datetime import datetime
        # start_time = datetime.now()

        if not self._ICP:
            creation_response = requests.post(href,
                                              params=params,
                                              headers=self._client._get_headers(content_type='application/zip'),
                                              data=data)
                                              # files=file_handler)
        else:
            creation_response = requests.post(href,
                                              params=params,
                                              headers=self._client._get_headers(content_type='application/zip'),
                                              data=data,
                                              # files=file_handler,
                                              verify=False)

        # end_time = datetime.now()
        #
        # dt = end_time - start_time
        # ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        #
        # print("Time taken to get response from import api call in ms: {}".format(ms))

        details = self._handle_response(expected_status_code=202,
                                        operationName=u'import start',
                                        response=creation_response)

        import_id = details[u'metadata']['id']

        print("import job with id {} has started. Monitor status using client.import_assets.get_details api. "
              "Check 'help(client.import_assets.get_details)' for details on the api usage".format(import_id))

        return details

    def _validate_input(self, meta_props):
        if 'name' not in meta_props:
            raise WMLClientError("Its mandatory to provide 'NAME' in meta_props. Example: "
                                 "client.import_assets.ConfigurationMetaNames.NAME: 'name'")

        if 'all_assets' not in meta_props and 'asset_ids' not in meta_props:
            raise WMLClientError("Its mandatory to provide either 'ALL_ASSETS' or 'ASSET_IDS' in meta_props. Example: "
                                 "client.import_assets.ConfigurationMetaNames.ALL_ASSETS: True")


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def cancel(self, import_id, space_id=None, project_id=None):
        """
            Cancel an import job. 'import_id' and 'space_id'( or 'project_id' has to be provided )
            Note: To delete an import_id job, use delete() api

            **Parameters**

            .. important::
                #. **import_id**:  Import job identifier\n
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

             >>> client.import_assets.cancel(import_id='6213cf1-252f-424b-b52d-5cdd9814956c',
             >>>                      space_id='3421cf1-252f-424b-b52d-5cdd981495fe')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        Import._validate_type(import_id, u'import_id', STR_TYPE, True)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        href = self._href_definitions.import_href(import_id)

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
                                        operationName=u'cancel import',
                                        response=cancel_response)

        if "SUCCESS" == details:
            print("Import job cancelled")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, import_id, space_id=None, project_id=None):
        """
            Deletes the given import_id job. 'space_id' or 'project_id' has to be provided

            **Parameters**

            .. important::
                #. **import_id**:  Import job identifier\n
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

             >>> client.import_assets.delete(import_id='6213cf1-252f-424b-b52d-5cdd9814956c',
             >>>                      space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        Import._validate_type(import_id, u'import_id', STR_TYPE, True)

        href = self._href_definitions.import_href(import_id)

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
                                        operationName=u'delete import job',
                                        response=delete_response)

        if "SUCCESS" == details:
            print("Import job deleted")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, import_id=None, space_id=None, project_id=None, limit=None):
        """
            Get metadata of the given import job. if no import_id is specified, all
            imports metadata is returned.

           **Parameters**

           .. important::
                #. **import_id**: import job identifier (optional)\n
                   **type**: str\n
                #. **space_id**:  Space identifier**\n
                   **type**: str\n
                #. **project_id**:  Project identifier**\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: import(s) metadata\n
                **return type**: dict\n
                dict (if import_id is not None) or {"resources": [dict]} (if import_id is None)\n

           .. note::
                If import_id is not specified, all import(s) metadata is fetched\n

           **Example**

             >>> details = client.import_assets.get_details(import_id)
             >>> details = client.import_assets.get_details()
         """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")
        
        Import._validate_type(import_id, u'import_id', STR_TYPE, False)
        Import._validate_type(limit, u'limit', int, False)

        href = self._href_definitions.imports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        return self._get_artifact_details(href, import_id, limit, 'import job', query_params=params)
    
    def list(self, space_id=None, project_id=None, limit=None):
        """
            List import jobs. If limit is set to None there will be only first 50 records shown.

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
                This method only prints the list of all import jobs in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.import_assets.list()
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
                   m[u'metadata'][u'created_at'],
                   m[u'entity'][u'status'][u'state']) for m in resources]

        self._list(values, [u'ID',  u'CREATED', u'STATUS'], limit, _DEFAULT_LIST_LENGTH)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(import_details):
        """
                Get ID of import job from import details

                **Parameters**

                .. important::

                   #. **import_details**:  Metadata of the import job\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: ID of the import job\n
                    **return type**: str\n

                **Example**

                 >>> id = client.import_assets.get_id(import_details)
        """
        Import._validate_type(import_details, u'import_details', object, True)

        return WMLResource._get_required_element_from_dict(import_details,
                                                           u'import_details',
                                                           [u'metadata', u'id'])

