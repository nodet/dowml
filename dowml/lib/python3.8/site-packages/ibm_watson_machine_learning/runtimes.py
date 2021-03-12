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
requests.packages.urllib3.disable_warnings()
import json
import os
from ibm_watson_machine_learning.utils import INSTANCE_DETAILS_TYPE, RUNTIME_SPEC_DETAILS_TYPE, MODEL_DETAILS_TYPE, LIBRARY_DETAILS_TYPE, FUNCTION_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, get_type_of_details, docstring_parameter, str_type_conv, print_text_header_h2, meta_props_str_conv
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.metanames import RuntimeMetaNames, LibraryMetaNames
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import MLRepositoryArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps, MetaNames
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS



def LibraryDefinition(name, version, filepath, description=None, platform=None, model_definition=None, custom=None, command=None, tags=None, space_uid=None):
    WMLResource._validate_type(name, 'name', STR_TYPE, True)
    WMLResource._validate_type(version, 'version', STR_TYPE, True)
    WMLResource._validate_type(platform, 'platform', dict, False)
    WMLResource._validate_type(description, 'description', STR_TYPE, False)
    WMLResource._validate_type(filepath, 'filepath', STR_TYPE, True)
    WMLResource._validate_type(model_definition, 'model_definition', bool, False)
    WMLResource._validate_type(custom, 'custom', dict, False)
    WMLResource._validate_type(command, 'command', STR_TYPE, False)
    WMLResource._validate_type(tags, 'tags', dict, False)
    WMLResource._validate_type(space_uid, 'space_uid', STR_TYPE, False)

    definition = {
        'name': name,
        'version': version,
        'filepath': filepath
    }

    if description is not None:
        definition['description'] = description

    if platform is not None:
        definition['platform'] = platform
    if model_definition is not None:
        definition['model_definition'] = model_definition
    if custom is not None:
        definition['custom'] = custom
    if command is not None:
        definition['command'] = command
    if tags is not None:
        definition['tags'] = tags
    if space_uid is not None:
        definition['space_uid'] = space_uid
    return definition


class Runtimes(WMLResource):
    """
        Creates Runtime Specs and associated Custom Libraries.

        .. note::
            There are a list of pre-defined runtimes available. To see the list of pre-defined runtimes, use:\n
            >>> client.runtimes.list(pre_defined=True)
    """
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.WSD and not client.CLOUD_PLATFORM_SPACES:
            Runtimes._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Runtimes._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self.ConfigurationMetaNames = RuntimeMetaNames()
        self.LibraryMetaNames = LibraryMetaNames()
        self._ICP = client.ICP

    def _create_library_from_definition(self, definition, runtime_definition):
        self._validate_meta_prop(definition, 'name', STR_TYPE, True)
        self._validate_meta_prop(definition, 'version', STR_TYPE, True)
        self._validate_meta_prop(definition, 'platform', dict, False)
        self._validate_meta_prop(definition, 'description', STR_TYPE, False)
        self._validate_meta_prop(definition, 'filepath', STR_TYPE, True)
        self._validate_meta_prop(definition, 'model_defintion', bool, False)
        self._validate_meta_prop(definition, 'custom', dict, False)
        self._validate_meta_prop(definition, 'command', STR_TYPE, False)
        self._validate_meta_prop(definition, 'tags', dict, False)
        self._validate_meta_prop(definition, 'space_uid', STR_TYPE, False)

        lib_metadata = {
            self.LibraryMetaNames.NAME: definition['name'],
            self.LibraryMetaNames.VERSION: definition['version'],
            self.LibraryMetaNames.PLATFORM:
                definition['platform']
                if 'platform' in definition and definition['platform'] is not None
                else {
                    "name": runtime_definition[self.ConfigurationMetaNames.PLATFORM]['name'],
                    "versions": [runtime_definition[self.ConfigurationMetaNames.PLATFORM]['version']]
                },
            self.LibraryMetaNames.FILEPATH: definition['filepath'],
            self.LibraryMetaNames.MODEL_DEFINITION: definition['model_definiton'],
            self.LibraryMetaNames.COMMAND: definition['command'],
            self.LibraryMetaNames.CUSTOM: definition['custom'],
            self.LibraryMetaNames.TAGS: definition['tags'],
            self.LibraryMetaNames.SPACE_UID: definition['space_uid'],
        }

        if 'description' in definition:
            lib_metadata[self.LibraryMetaNames.DESCRIPTION] = definition['description']
        if 'tags' in definition:
            lib_metadata[self.LibraryMetaNames.TAGS] = definition['tags']
        if 'model_definition' in definition:
            lib_metadata[self.LibraryMetaNames.MODEL_DEFINITION] = definition['model_definition']
        if 'custom' in definition:
            lib_metadata[self.LibraryMetaNames.CUSTOM] = definition['custom']
        if 'command' in definition:
            lib_metadata[self.LibraryMetaNames.COMMAND] = definition['command']
        if 'space_uid' in definition:
            lib_metadata[self.LibraryMetaNames.SPACE_UID] = definition['space_uid']

        details = self.store_library(lib_metadata)
        return self.get_library_uid(details)

    def store_library(self, meta_props):
        """
               Create a library.\n

               **Parameters**

               .. important::

                    #. **meta_props**:  meta data of the libraries configuration. To see available meta names use:\n
                                        >>> client.runtimes.LibraryMetaNames.get()

                       **type**: dict\n

               **Output**

               .. important::

                    **returns**: Metadata of the library created.\n
                    **return type**: dict\n

               **Example**

                >>> library_details = client.runtimes.store_library({
                >>> client.runtimes.LibraryMetaNames.NAME: "libraries_custom",
                >>> client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
                >>> client.runtimes.LibraryMetaNames.FILEPATH: custom_library_path,
                >>> client.runtimes.LibraryMetaNames.VERSION: "1.0",
                >>> client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]}
                >>> })
        """

        if self._client.ICP_30:
           print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        self.LibraryMetaNames._validate(meta_props)

        lib_metadata = self.LibraryMetaNames._generate_resource_metadata(meta_props, with_validation=True)
        if self._client.CAMS:
            if self._client.default_space_id is not None:
                lib_metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
            else:
                raise WMLClientError(
                    "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")

        try:
            if not self._ICP:
                response_post = requests.post(self._href_definitions.get_custom_libraries_href(), json=lib_metadata,
                                              headers=self._client._get_headers())
            else:
                response_post = requests.post(self._href_definitions.get_custom_libraries_href(), json=lib_metadata,
                                              headers=self._client._get_headers(), verify=False)
            details = self._handle_response(201, u'saving libraries', response_post)

            if self.LibraryMetaNames.FILEPATH in meta_props:
                try:
                    base_url = self._wml_credentials[u'url']
                    libraries_content_url = base_url + details[u'metadata'][u'href'] + '/content'

                    put_header = self._client._get_headers(no_content_type=True)
                    with open(meta_props[self.LibraryMetaNames.FILEPATH], 'rb') as data:
                        if not self._ICP:
                            response_definition_put = requests.put(libraries_content_url, data=data, headers=put_header)
                        else:
                            response_definition_put = requests.put(libraries_content_url, data=data, headers=put_header,
                                                                   verify=False)

                except Exception as e:
                    raise e
                self._handle_response(200, u'saving libraries content', response_definition_put, False)
        except Exception as e:
            raise WMLClientError('Failure during creation of libraries.', e)
        return details

    def _create_runtime_spec(self, custom_libs_list, meta_props):

        metadata = {
            self.ConfigurationMetaNames.NAME : meta_props[self.ConfigurationMetaNames.NAME],
            self.ConfigurationMetaNames.PLATFORM: meta_props[self.ConfigurationMetaNames.PLATFORM],
        }

        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            metadata[self.ConfigurationMetaNames.DESCRIPTION] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

        if self.ConfigurationMetaNames.CUSTOM in meta_props:
            metadata[self.ConfigurationMetaNames.CUSTOM] = {meta_props[self.ConfigurationMetaNames.CUSTOM]}

        if self.ConfigurationMetaNames.COMPUTE in meta_props:
            metadata[self.ConfigurationMetaNames.COMPUTE] = meta_props[self.ConfigurationMetaNames.COMPUTE]

        if self.ConfigurationMetaNames.SPACE_UID in meta_props:
            metadata[self.ConfigurationMetaNames.SPACE_UID] = {
                "href": "/v4/spaces/"+meta_props[self.ConfigurationMetaNames.SPACE_UID]
            }
        if self._client.CAMS:
            if self._client.default_space_id is not None:
                metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
            else:
                raise WMLClientError(
                    "It is mandatory to set the space. Use client.set.default_space(<SPACE_UID>) to proceed.")

        if custom_libs_list is not None:
            custom_list = []
            for uid in custom_libs_list:
                each_href = {"href": "/v4/libraries/" + uid}
                custom_list.append(each_href)
            metadata["custom_libraries"] = custom_list

        if self.ConfigurationMetaNames.CONFIGURATION_FILEPATH in meta_props:
            metadata[MetaNames.CONTENT_LOCATION] = meta_props[self.ConfigurationMetaNames.CONFIGURATION_FILEPATH]

        try:
            if not self._ICP:
                response_post = requests.post(self._href_definitions.get_runtimes_href(), json=metadata,
                                              headers=self._client._get_headers())
            else:
                response_post = requests.post(self._href_definitions.get_runtimes_href(), json=metadata,
                                              headers=self._client._get_headers(), verify=False)
            details = self._handle_response(201, u'saving runtimes', response_post)
            if self.ConfigurationMetaNames.CONFIGURATION_FILEPATH in meta_props:
                try:
                    runtimes_content_url = self._wml_credentials[u'url'] + details[u'metadata'][u'href'] + '/content'

                    put_header = self._client._get_headers(content_type="text/plain")
                    with open(meta_props[self.ConfigurationMetaNames.CONFIGURATION_FILEPATH], 'rb') as data:
                        if not self._ICP:
                            response_definition_put = requests.put(runtimes_content_url, data=data,headers=put_header)
                        else:
                            response_definition_put = requests.put(runtimes_content_url, data=data,headers=put_header,
                                                                   verify=False)

                except Exception as e:
                    raise e
                self._handle_response(200, u'saving runtimes content', response_definition_put, False)
        except Exception as e:
            raise WMLClientError('Failure during creation of runtime.', e)
        return details['metadata']['guid']


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, meta_props):
        """
               Create a runtime.\n

               **Parameters**

               .. important::

                    #. **meta_props**:  meta data of the runtime configuration. To see available meta names use:\n
                                        >>> client.runtimes.ConfigurationMetaNames.get()

                       **type**: dict\n

               **Output**

               .. important::

                    **returns**: Metadata of the runtime created\n
                    **return type**: dict\n

               **Example**

                Creating a library\n

                >>> lib_meta = {
                >>> client.runtimes.LibraryMetaNames.NAME: "libraries_custom",
                >>> client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
                >>> client.runtimes.LibraryMetaNames.FILEPATH: "/home/user/my_lib.zip",
                >>> client.runtimes.LibraryMetaNames.VERSION: "1.0",
                >>> client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]}
                >>> }
                >>> custom_library_details = client.runtimes.store_library(lib_meta)
                >>> custom_library_uid = client.runtimes.get_library_uid(custom_library_details)

                Creating a runtime\n

                >>> runtime_meta = {
                >>> client.runtimes.ConfigurationMetaNames.NAME: "runtime_spec_python_3.5",
                >>> client.runtimes.ConfigurationMetaNames.DESCRIPTION: "test",
                >>> client.runtimes.ConfigurationMetaNames.PLATFORM: {
                >>> "name": "python",
                >>>  "version": "3.5"
                >>> },
                >>> client.runtimes.ConfigurationMetaNames.LIBRARIES_UIDS: [custom_library_uid] # already existing lib is linked here
                >>> }
                >>> runtime_details = client.runtimes.store(runtime_meta)

        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        self.ConfigurationMetaNames._validate(meta_props)

        custom_libs_list = []

        # if self.ConfigurationMetaNames.LIBRARIES_DEFINITIONS in meta_props:
        #     custom_libs_list.extend(
        #         [self._create_library_from_definition(definition, meta_props) for definition in
        #          meta_props[self.ConfigurationMetaNames.LIBRARIES_DEFINITIONS]]
        #     )

        if self.ConfigurationMetaNames.LIBRARIES_UIDS in meta_props:
            custom_libs_list.extend(meta_props[self.ConfigurationMetaNames.LIBRARIES_UIDS])

        runtime_uid = self._create_runtime_spec(custom_libs_list, meta_props)

        return self.get_details(runtime_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, runtime_uid=None,pre_defined=False, limit=None):
        """
           Get metadata of stored runtime(s). If runtime UID is not specified returns all runtimes metadata.

           **Parameters**

           .. important::
                #. **runtime_uid**: runtime UID (optional)\n
                   **type**: str\n
                #. **pre_defined**:  Boolean indicating to display predefined runtimes only. Default value is set to 'False'\n
                   **type**: bool\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of runtime(s)\n
                **return type**: dict
                The output can be {"resources": [dict]} or a dict\n

           .. note::
                If UID is not specified, all runtimes metadata is fetched\n

           **Example**

            >>> runtime_details = client.runtimes.get_details(runtime_uid)
            >>> runtime_details = client.runtimes.get_details(runtime_uid=runtime_uid)
            >>> runtime_details = client.runtimes.get_details()
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        runtime_uid = str_type_conv(runtime_uid)
        Runtimes._validate_type(runtime_uid, u'runtime_uid', STR_TYPE, False)

        # if runtime_uid is not None and not is_uid(runtime_uid):
        #     raise WMLClientError(u'\'runtime_uid\' is not an uid: \'{}\''.format(runtime_uid))

        url = self._href_definitions.get_runtimes_href()
        if runtime_uid is not None or self._client.default_project_id is not None:
         return self._get_no_space_artifact_details(url, runtime_uid, limit, 'runtime specs', pre_defined="True")
        if pre_defined:
         return self._get_artifact_details(url, runtime_uid, limit, 'runtime specs',pre_defined="True")
        else:
         return self._get_artifact_details(url, runtime_uid, limit, 'runtime specs')

    def get_library_details(self, library_uid=None, limit=None):
        """
           Get metadata of stored librarie(s). If library UID is not specified returns all libraries metadata.

           **Parameters**

           .. important::
                #. **library_uid**: library UID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of library(s)\n
                **return type**: dict
                The output can be {"resources": [dict]} or a dict\n

           .. note::
                If UID is not specified, all libraries metadata is fetched\n

           **Example**

            >>> library_details = client.runtimes.get_library_details(library_uid)
            >>> library_details = client.runtimes.get_library_details(library_uid=library_uid)
            >>> library_details = client.runtimes.get_library_details()
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        library_uid = str_type_conv(library_uid)
        Runtimes._validate_type(library_uid, u'library_uid', STR_TYPE, False)

        if library_uid is not None and not is_uid(library_uid):
            raise WMLClientError(u'\'library_uid\' is not an uid: \'{}\''.format(library_uid))

        url = self._href_definitions.get_custom_libraries_href()
        if library_uid is not None or self._client.default_project_id is not None:
            return self._get_no_space_artifact_details(url, library_uid, limit, 'libraries')
        return self._get_artifact_details(url, library_uid, limit, 'libraries')

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(details):
        """
            Get runtime_href from runtime details.

            **Parameters**

            .. important::
                #. **runtime_details**:  Metadata of the runtime\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: runtime href\n
                **return type**: str

            **Example**

             >>> runtime_details = client.runtimes.get_details(runtime_uid)
             >>> runtime_href = client.runtimes.get_href(runtime_details)
        """

        Runtimes._validate_type(details, u'details', dict, True)
        Runtimes._validate_type_of_details(details, [RUNTIME_SPEC_DETAILS_TYPE, MODEL_DETAILS_TYPE, FUNCTION_DETAILS_TYPE])

        details_type = get_type_of_details(details)

        if details_type == RUNTIME_SPEC_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'runtime_details', ['metadata', 'href'])
        elif details_type == MODEL_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'model_details', ['entity', 'runtime', 'href'])
        elif details_type == FUNCTION_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'function_details', ['entity', 'runtime', 'href'])
        else:
            raise WMLClientError('Unexpected details type: {}'.format(details_type))

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(details):
        """
            Get runtime_uid from runtime details.

            **Parameters**

            .. important::
                #. **runtime_details**:  Metadata of the runtime\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: runtime UID\n
                **return type**: str

            **Example**

             >>> runtime_details = client.runtimes.get_details(runtime_uid)
             >>> runtime_uid = client.runtimes.get_uid(runtime_details)
        """

        Runtimes._validate_type(details, u'details', dict, True)
        Runtimes._validate_type_of_details(details, [RUNTIME_SPEC_DETAILS_TYPE, MODEL_DETAILS_TYPE, FUNCTION_DETAILS_TYPE])

        details_type = get_type_of_details(details)

        if details_type == RUNTIME_SPEC_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'runtime_details', ['metadata', 'guid'])
        elif details_type == MODEL_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'model_details', ['entity', 'runtime', 'href']).split('/')[-1]
        elif details_type == FUNCTION_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'function_details', ['entity', 'runtime', 'href']).split('/')[-1]
        else:
            raise WMLClientError('Unexpected details type: {}'.format(details_type))

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_library_href(library_details):
        """
            Get library_href from library details.

            **Parameters**

            .. important::
                #. **library_details**:  Metadata of the library\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: library href\n
                **return type**: str

            **Example**

             >>> library_details = client.runtimes.get_library_details(library_uid)
             >>> library_url = client.runtimes.get_library_href(library_details)
        """

        Runtimes._validate_type(library_details, u'library_details', dict, True)
        Runtimes._validate_type_of_details(library_details, LIBRARY_DETAILS_TYPE)

        return Runtimes._get_required_element_from_dict(library_details, 'library_details', ['metadata', 'href'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_library_uid(library_details):
        """
            Get library_uid from library details.

            **Parameters**

            .. important::
                #. **library_details**:  Metadata of the library\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: library UID\n
                **return type**: str

            **Example**

             >>> library_details = client.runtimes.get_library_details(library_uid)
             >>> library_uid = client.runtimes.get_library_uid(library_details)
        """

        Runtimes._validate_type(library_details, u'library_details', dict, True)
        Runtimes._validate_type_of_details(library_details, LIBRARY_DETAILS_TYPE)

        # TODO error handling
        return Runtimes._get_required_element_from_dict(library_details, 'library_details', ['metadata', 'guid'])

    def _get_runtimes_uids_for_lib(self, library_uid, runtime_details=None):
        # Return list of runtimes which contains library_uid that is passed.
        if runtime_details is None:
            runtime_details = self.get_details()

        return list(map(
            lambda x: x['metadata']['guid'],
            filter(
                lambda x: any(
                    filter(
                        lambda y: library_uid in y['href'],
                        x['entity']['custom_libraries'] if 'custom_libraries' in x['entity'] else [])
                ),
                runtime_details['resources']
            )
        ))

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, runtime_uid, with_libraries=False):
        """
            Delete a runtime.

            **Parameters**

            .. important::
                #. **runtime_uid**:  Runtime UID\n
                   **type**: str\n
                #. **with_libraries**:  Boolean value indicating an option to delete the libraries associated with the runtime\n
                   **type**: bool\n


            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.runtimes.delete(deployment_uid)
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        runtime_uid = str_type_conv(runtime_uid)
        Runtimes._validate_type(runtime_uid, u'runtime_uid', STR_TYPE, True)
        Runtimes._validate_type(with_libraries, u'autoremove', bool, True)

        if runtime_uid is not None and not is_uid(runtime_uid):
            raise WMLClientError(u'\'runtime_uid\' is not an uid: \'{}\''.format(runtime_uid))

        if with_libraries:
            runtime_details = self.get_details(runtime_uid)

        url = self._href_definitions.get_runtime_href(runtime_uid)

        if not self._ICP:
            response_delete = requests.delete(
                url,
                headers=self._client._get_headers())
        else:
            response_delete = requests.delete(
                url,
                headers=self._client._get_headers(),
                verify=False)


        if with_libraries:
            if 'custom_libraries' in runtime_details['entity']:
                details = self.get_details()
                custom_libs_uids = map(lambda x: x['href'].split('/')[-1], runtime_details['entity']['custom_libraries'])
                custom_libs_to_remove = filter(
                    lambda x: len(self._get_runtimes_uids_for_lib(x, details)) == 0,
                    custom_libs_uids
                )

                for uid in custom_libs_to_remove:
                    print('Deleting orphaned library \'{}\' during autoremove delete.'.format(uid))
                    delete_status = self.delete_library(uid)
                    print(delete_status)
        return self._handle_response(204, u'runtime deletion', response_delete, False)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _delete_orphaned_libraries(self):
        """
            Delete all custom libraries without runtime.

            A way you might use me is:

            >>> client.runtimes.delete_orphaned_libraries()
        """
        lib_details = self.get_library_details()
        details = self.get_details()
        for lib in lib_details['resources']:
            lib_uid = lib['metadata']['guid']
            if len(self._get_runtimes_uids_for_lib(lib_uid, details)) == 0:
                print('Deleting orphaned \'{}\' library... '.format(lib_uid), end="")
                library_endpoint = self._href_definitions.get_custom_library_href(lib_uid)
                if not self._ICP:
                    response_delete = requests.delete(library_endpoint, headers=self._client._get_headers())
                else:
                    response_delete = requests.delete(library_endpoint, headers=self._client._get_headers(), verify=False)

                try:
                    self._handle_response(204, u'library deletion', response_delete, False)
                    print('SUCCESS')
                except:
                    pass


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete_library(self, library_uid):
        """
            Delete a library.

            **Parameters**

            .. important::
                #. **library_uid**:  Library UID\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client.runtimes.delete_library(library_uid)
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        Runtimes._validate_type(library_uid, u'library_uid', STR_TYPE, True)
        library_endpoint = self._href_definitions.get_custom_library_href(library_uid)
        if not self._ICP:
            response_delete = requests.delete(library_endpoint, headers=self._client._get_headers())
        else:
            response_delete = requests.delete(library_endpoint, headers=self._client._get_headers(), verify=False)

        return self._handle_response(204, u'library deletion', response_delete, False)

    def list(self, limit=None,pre_defined=False):
        """
           List stored runtimes. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n
                #. **pre_defined**:  Boolean indicating to display predefined runtimes only. Default value is set to 'False'\n
                   **type**: bool\n

           **Output**

           .. important::
                This method only prints the list of runtimes in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.runtimes.list()
            >>> client.runtimes.list(pre_defined=True)
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        details = self.get_details(pre_defined=pre_defined)
        resources = details[u'resources']
        values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], m[u'entity'][u'platform']['name']) for m in resources]

        self._list(values, [u'GUID', u'NAME', u'CREATED', u'PLATFORM'], limit, 50)

    def _list_runtimes_for_libraries(self): # TODO make public when the time'll come
        """
           List runtimes uids for libraries.

           A way you might use me is:

           >>> client.runtimes.list_runtimes_for_libraries()
           >>> client.runtimes.list_runtimes_for_libraries(library_uid)
        """
        details = self.get_library_details()
        runtime_details = self.get_details()

        values = [
            (m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'version'],
             ', '.join(self._get_runtimes_uids_for_lib(m[u'metadata'][u'guid'], runtime_details))) for m in
            details['resources']]

        values = sorted(sorted(values, key=lambda x: x[2], reverse=True), key=lambda x: x[1])

        from tabulate import tabulate

        header = [u'GUID', u'NAME', u'VERSION', u'RUNTIME SPECS']
        table = tabulate([header] + values)

        print(table)

    def list_libraries(self, runtime_uid=None, limit=None):
        """
           List stored libraries. If runtime UID is not provided, all libraries are listed else, libraries associated with a runtime are listed. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **runtime_uid**:  runtime UID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of libraries in a table format.\n
                **return type**: None\n


           **Example**

            >>> client.runtimes.list_libraries()
            >>> client.runtimes.list_libraries(runtime_uid)
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        runtime_uid = str_type_conv(runtime_uid)
        Runtimes._validate_type(runtime_uid, u'runtime_uid', STR_TYPE, False)

        if runtime_uid is None:
            details = self.get_library_details()

            resources = details[u'resources']
            values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'version'], m[u'metadata'][u'created_at'],
                       m[u'entity'][u'platform']['name'], m[u'entity'][u'platform'][u'versions']) for m in
                      resources]

            self._list(values, [u'GUID', u'NAME', u'VERSION', u'CREATED', u'PLATFORM NAME', u'PLATFORM VERSIONS'], limit, 50)
        else:
            details = self.get_details(runtime_uid)

            if 'custom_libraries' not in details['entity'] or len(details['entity']['custom_libraries']) == 0:
                print('No libraries found for this runtime.')
                return

            values = [(m[u'href'].split('/')[-1], u'') for m in details['entity']['custom_libraries']]


            from tabulate import tabulate

            header = [u'GUID']
            table = tabulate([header] + values)

            print(table)

    def download_configuration(self, runtime_uid, filename=None):
        """
                Downloads configuration file for runtime with specified uid.

                **Parameters**

                .. important::

                    #. **runtime_uid**:  UID of runtime\n
                       **type**: str\n
                    #. **filename**:  filename of downloaded archive (optional)\n
                       **default value**: runtime_configuration.yaml\n
                       **type**: str\n


                **Output**

                .. important::

                    **returns**: Path to the downloaded runtime configuration\n
                    **return type**: str\n

                .. note::

                   If filename is not specified, the default filename is "runtime_configuration.yaml".\n

                **Example**

                   >>> filename="runtime.yml"
                   >>> client.runtimes.download_configuration(runtime_uid, filename=filename)
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        runtime_uid = str_type_conv(runtime_uid)
        Runtimes._validate_type(runtime_uid, u'runtime_uid', STR_TYPE, True)

        if not is_uid(runtime_uid):
            raise WMLClientError(u'\'runtime_uid\' is not an uid: \'{}\''.format(runtime_uid))

        download_url = self._href_definitions.get_runtime_href(runtime_uid) + '/content'

        if not self._ICP:
            response_get = requests.get(
                download_url,
                headers=self._client._get_headers())
        else:
            response_get = requests.get(
                download_url,
                headers=self._client._get_headers(),
                verify=False)

        if filename is None:
            filename = 'runtime_configuration.yaml'

        if response_get.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response_get.content)
                new_file.close()

                print(u'Successfully downloaded runtime configuration file: ' + str(filename))
                return os.getcwd() + "/"+filename
        else:
            if response_get.status_code == 404 and "content_does_not_exist" in str(response_get.text):
                raise WMLClientError(u'Unable to download configuration. download configruation can be invoked '
                                     u'only when CONFIGURATION_FILEPATH meta prop is set during store. ')
            raise WMLClientError(u'Unable to download configuration content: ' + response_get.text)

    def download_library(self, library_uid, filename=None):
        """
                Downloads library content with specified uid.

                **Parameters**

                .. important::

                      #. **library_uid**:  UID of library\n
                         **type**: str\n
                      #. **filename**:  filename of downloaded archive (optional)\n
                         **default value**: <LIBRARY-NAME>-<LIBRARY-VERSION>.zip\n
                         **type**: str\n

                **Output**

                .. important::

                       **returns**: Path to the downloaded library content\n
                       **return type**: str\n

                .. note::

                    If filename is not specified, the default filename is "<LIBRARY-NAME>-<LIBRARY-VERSION>.zip".\n

                **Example**

                    >>> filename="library.tgz"
                    >>> client.runtimes.download_library(runtime_uid, filename=filename)
        """

        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        library_uid = str_type_conv(library_uid)
        Runtimes._validate_type(library_uid, u'library_uid', STR_TYPE, True)

        if not is_uid(library_uid):
            raise WMLClientError(u'\'library_uid\' is not an uid: \'{}\''.format(library_uid))

        download_url = self._href_definitions.get_custom_library_href(library_uid) + '/content'

        if not self._ICP:
            response_get = requests.get(
                download_url,
                headers=self._client._get_headers())
        else:
            response_get = requests.get(
                download_url,
                headers=self._client._get_headers(),
                verify=False)

        if filename is None:
            details = self.get_library_details(library_uid)
            filename = '{}-{}.zip'.format(details['entity']['name'], details['entity']['version'])

        if response_get.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response_get.content)
                new_file.close()

                print(u'Successfully downloaded library content: ' + str(filename))
                return os.getcwd() + "/"+filename
        else:
            raise WMLClientError(u'Unable to download library content: ' + response_get.text)

    def update_library(self, library_uid, changes):
        """
                Updates existing library metadata.

                **Parameters**

                .. important::
                    #. **library_uid**:  UID of library which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of updated library\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.runtimes.LibraryMetaNames.NAME:"updated_lib"
                 >>> }
                 >>> library_details = client.runtimes.update_library(library_uid, changes=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        library_uid = str_type_conv(library_uid)
        self._validate_type(library_uid, u'library_uid', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self.get_library_details(library_uid)

        patch_payload = self.LibraryMetaNames._generate_patch_payload(details['entity'], changes, with_validation=True)

        url = self._href_definitions.get_custom_library_href(library_uid)
        if not self._ICP:
            response = requests.patch(url, json=patch_payload, headers=self._client._get_headers())
        else:
            response = requests.patch(url, json=patch_payload, headers=self._client._get_headers(), verify=False)
        updated_details = self._handle_response(200, u'library patch', response)

        return updated_details

    def update_runtime(self, runtime_uid, changes):
        """
                Updates existing runtime metadata.

                **Parameters**

                .. important::
                    #. **runtime_uid**:  UID of runtime which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of updated runtime\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.runtimes.ConfigurationMetaNames.NAME:"updated_runtime"
                 >>> }
                 >>> runtime_details = client.runtimes.update(runtime_uid, changes=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        library_uid = str_type_conv(runtime_uid)
        self._validate_type(library_uid, u'runtime_uid', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self.get_details(runtime_uid)

        patch_payload = self.LibraryMetaNames._generate_patch_payload(details['entity'], changes, with_validation=True)

        url = self._href_definitions.get_runtime_href(runtime_uid)
        if not self._ICP:
            response = requests.patch(url, json=patch_payload, headers=self._client._get_headers())
        else:
            response = requests.patch(url, json=patch_payload, headers=self._client._get_headers(), verify=False)
        updated_details = self._handle_response(200, u'library patch', response)

        return updated_details

    def clone_runtime(self, runtime_uid, space_id=None, action="copy", rev_id=None):
        """
                Creates a new runtime identical with the given runtime either in the same space or in a new space. All dependent assets will be cloned too.

                **Parameters**

                .. important::
                    #. **model_id**:  Guid of the runtime to be cloned:\n

                       **type**: str\n

                    #. **space_id**: Guid of the space to which the runtime needs to be cloned. (optional)

                       **type**: str\n

                    #. **action**: Action specifying "copy" or "move". (optional)

                       **type**: str\n

                    #. **rev_id**: Revision ID of the runtime. (optional)

                       **type**: str\n

                **Output**

                .. important::

                        **returns**: Metadata of the runtime cloned.\n
                        **return type**: dict\n

                **Example**
                 >>> client.runtimes.clone_runtime(runtime_uid=artifact_id,space_id=space_uid,action="copy")

                .. note::
                    * If revision id is not specified, all revisions of the artifact are cloned\n

                    * Default value of the parameter action is copy\n

                    * Space guid is mandatory for move action\n

        """
        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        artifact = str_type_conv(runtime_uid)
        Runtimes._validate_type(artifact, 'runtime_uid', STR_TYPE, True)
        space = str_type_conv(space_id)
        rev = str_type_conv(rev_id)
        action = str_type_conv(action)
        clone_meta = {}
        if space is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space}
        if action is not None:
            clone_meta["action"] = action
        if rev is not None:
            clone_meta["rev"] = rev

        url = self._href_definitions.get_runtime_href(runtime_uid)
        if not self._ICP:
            response_post = requests.post(url, json=clone_meta,
                                              headers=self._client._get_headers())
        else:
            response_post = requests.post(url, json=clone_meta,
                                              headers=self._client._get_headers(), verify=False)

        details = self._handle_response(expected_status_code=200, operationName=u'cloning runtime',
                                            response=response_post)

        return details
    def clone_library(self, library_uid, space_id=None, action="copy", rev_id=None):
        """
                Creates a new function library with the given library either in the same space or in a new space. All dependent assets will be cloned too.

                **Parameters**

                .. important::
                    #. **model_id**:  Guid of the library to be cloned:\n

                       **type**: str\n

                    #. **space_id**: Guid of the space to which the library needs to be cloned. (optional)

                       **type**: str\n

                    #. **action**: Action specifying "copy" or "move". (optional)

                       **type**: str\n

                    #. **rev_id**: Revision ID of the library. (optional)

                       **type**: str\n

                **Output**

                .. important::

                        **returns**: Metadata of the library cloned.\n
                        **return type**: dict\n

                **Example**
                 >>> client.runtmes.clone_library(library_uid=artifact_id,space_id=space_uid,action="copy")

                .. note::
                    * If revision id is not specified, all revisions of the artifact are cloned\n

                    * Default value of the parameter action is copy\n

                    * Space guid is mandatory for move action\n

        """
        if self._client.ICP_30:
            print("WARNING!! 'runtimes' is DEPRECATED. Use 'software_specifications' instead to create and manage runtimes/specifications" )

        artifact = str_type_conv(library_uid)
        Runtimes._validate_type(artifact, 'library_uid', STR_TYPE, True)
        space = str_type_conv(space_id)
        rev = str_type_conv(rev_id)
        action = str_type_conv(action)
        clone_meta = {}
        if space is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space}
        if action is not None:
            clone_meta["action"] = action
        if rev is not None:
            clone_meta["rev"] = rev

        url = self._href_definitions.get_custom_library_href(library_uid)
        if not self._ICP:
            response_post = requests.post(url, json=clone_meta,
                                              headers=self._client._get_headers())
        else:
            response_post = requests.post(url, json=clone_meta,
                                              headers=self._client._get_headers(), verify=False)

        details = self._handle_response(expected_status_code=200, operationName=u'cloning library',
                                            response=response_post)

        return details


