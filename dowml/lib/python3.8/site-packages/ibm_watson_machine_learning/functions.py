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
from ibm_watson_machine_learning.utils import INSTANCE_DETAILS_TYPE, FUNCTION_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, str_type_conv, is_of_python_basic_type, meta_props_str_conv
from ibm_watson_machine_learning.metanames import FunctionMetaNames, FunctionNewMetaNames
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS
_DEFAULT_LIST_LENGTH = 50


class Functions(WMLResource):
    """
    Store and manage your functions.
    """
    # ConfigurationMetaNames = FunctionMetaNames()
    """MetaNames for python functions creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.WSD and not client.CLOUD_PLATFORM_SPACES and not client.ICP_PLATFORM_SPACES:
            Functions._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Functions._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self._ICP = client.ICP
        self._WSD = client.WSD

        if client.CLOUD_PLATFORM_SPACES or client.ICP_PLATFORM_SPACES:
            self.ConfigurationMetaNames = FunctionNewMetaNames()
        else:
            self.ConfigurationMetaNames = FunctionMetaNames()

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store(self, function, meta_props):
        """
            Create a function.

            As a 'function' may be used one of the following:
             - filepath to gz file
             - 'score' function reference, where the function is the function which will be deployed
             - generator function, which takes no argument or arguments which all have primitive python default values and as result return 'score' function

           **Parameters**

           .. important::
                #. **meta_props**:  meta data or name of the function. To see available meta names use **client._functions.ConfigurationMetaNames.show()**\n
                   **type**: str or dict\n
                #. **function**:  path to file with archived function content or function (as described above)\n
                   **type**: str or function\n

           **Output**

           .. important::
                **returns**: stored function metadata\n
                **return type**: dict\n

           **Example**

            The most simple use is (using `score` function):

             >>> meta_props = {
             >>>    client._functions.ConfigurationMetaNames.NAME: "function",
             >>>    client._functions.ConfigurationMetaNames.DESCRIPTION: "This is ai function",
             >>>    client._functions.ConfigurationMetaNames.SOFTWARE_SPEC_UID: "53dc4cf1-252f-424b-b52d-5cdd9814987f"}

             >>> def score(payload):
             >>>      values = [[row[0]*row[1]] for row in payload['values']]
             >>>      return {'fields': ['multiplication'], 'values': values}
             >>> stored_function_details = client._functions.store(score, meta_props)

             Other, more interesting example is using generator function.
             In this situation it is possible to pass some variables:

                >>> wml_creds = {...}
                >>> def gen_function(wml_credentials=wml_creds, x=2):
                        def f(payload):
                            values = [[row[0]*row[1]*x] for row in payload['values']]
                            return {'fields': ['multiplication'], 'values': values}
                        return f
                >>> stored_function_details = client._functions.store(gen_function, meta_props)
            """
        WMLResource._chk_and_block_create_update_for_python36(self)
        self._client._check_if_either_is_set()

        function = str_type_conv(function)
        import types
        Functions._validate_type(function, u'function', [STR_TYPE, types.FunctionType], True)
        meta_props = str_type_conv(meta_props)  # meta_props may be str, in this situation for py2 it will be converted to unicode
        Functions._validate_type(meta_props, u'meta_props', [dict, STR_TYPE], True)

        if type(meta_props) is STR_TYPE:
            meta_props = {
                self.ConfigurationMetaNames.NAME: meta_props
            }

        self.ConfigurationMetaNames._validate(meta_props)
        if self._client.WSD:
            if "space_uid" in meta_props:
                raise WMLClientError(u'Invalid input SPACE_UID in meta_props. SPACE_UID not supported for Watson Studio Desktop.')
        user_content_file = False
        if type(function) is STR_TYPE:
            content_path = function
            user_content_file = True
        else:
            try:
                import inspect
                import gzip
                import uuid
                import re
                import shutil
                code = inspect.getsource(function).split('\n')
                r = re.compile(r"^ *")
                m = r.search(code[0])
                intend = m.group(0)

                code = [line.replace(intend, '', 1) for line in code]

                args_spec = inspect.getargspec(function)

                defaults = args_spec.defaults if args_spec.defaults is not None else []
                args = args_spec.args if args_spec.args is not None else []

                if function.__name__ is 'score':
                    code = '\n'.join(code)
                    file_content = code
                elif len(args) == len(defaults):
                    for i, d in enumerate(defaults):
                        if not is_of_python_basic_type(d):
                            raise UnexpectedType(args[i], 'primitive python type', type(d))

                    new_header = 'def {}({}):'.format(
                        function.__name__,
                        ', '.join(
                            ['{}={}'.format(arg_name, json.dumps(default)) for arg_name, default in zip(args, defaults)]
                        )
                    )

                    code[0] = new_header
                    code = '\n'.join(code)
                    file_content = """
{}

score = {}()
""".format(code, function.__name__)
                else:
                    raise WMLClientError("Function passed is not \'score\' function nor generator function. Generator function should have no arguments or all arguments with primitive python default values.")

                tmp_uid = 'tmp_python_function_code_{}'.format(str(uuid.uuid4()).replace('-', '_'))
                filename = '{}.py'.format(tmp_uid)

                with open(filename, 'w') as f:
                    f.write(file_content)

                archive_name = '{}.py.gz'.format(tmp_uid)

                with open(filename, 'rb') as f_in:
                    with gzip.open(archive_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(filename)

                content_path = archive_name
            except Exception as e:
                try:
                    os.remove(filename)
                except:
                    pass
                try:
                    os.remove(archive_name)
                except:
                    pass
                raise WMLClientError('Exception during getting function code.', e)

        try:

            if self._client.WSD:
                if self._client.WSD_20:
                    if self.ConfigurationMetaNames.RUNTIME_UID not in meta_props and self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props:
                        raise WMLClientError("Invalid input. It is mandatory to provide RUNTIME_UID or "
                                             "SOFTWARE_SPEC_UID in meta_props. RUNTIME_UID is deprecated")
                else:
                    if self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                        raise WMLClientError('Missing RUNTIME_UID in input meta_props.')
                if self.ConfigurationMetaNames.TYPE not in meta_props:
                    meta_props.update({self.ConfigurationMetaNames.TYPE: 'python'})

                import copy
                function_metadata = self.ConfigurationMetaNames._generate_resource_metadata(meta_props,
                                                                                            with_validation=True,
                                                                                           client=self._client)
                input_schemas = []
                output_schemas = []
                if self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS in meta_props and \
                        meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS] is not None:
                    input_schemas = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS]
                    function_metadata.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS)
                if self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS in meta_props and \
                        meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS] is not None:
                    output_schemas = meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS]
                    function_metadata.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS)
                if len(input_schemas) != 0 or len(output_schemas) != 0:
                    function_metadata.update({"schemas": {
                        "input": input_schemas,
                        "output": output_schemas}
                    })
                if self._client.WSD_20:
                    if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props and \
                            meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID] is not None:
                        function_metadata.pop('software_spec')
                        function_metadata.update(
                            {"software_spec": {"base_id":meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}})


                if self._client.default_project_id is not None:
                    function_metadata['project'] = {'href': "/v2/projects/" + self._client.default_project_id}
                import copy
                payload = copy.deepcopy(function_metadata)
                # with open(content_path, 'rb') as datafile:
                #     data = datafile.read()
                details = Functions._wsd_create_asset(self, "wml_function", payload, meta_props, content_path,user_content_file)
            else:
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    if self.ConfigurationMetaNames.RUNTIME_UID not in meta_props and self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props:
                        import sys
                        # print('No RUNTIME_UID passed. Creating default runtime... ', end="")
                        # meta = {
                        #     self._client.runtimes.ConfigurationMetaNames.NAME: meta_props[self.ConfigurationMetaNames.NAME] + "_py_3.5",
                        #     self._client.runtimes.ConfigurationMetaNames.PLATFORM: {
                        #         "name": "python",
                        #         "version": float(sys.version.split()[0][0:3])
                        #     }
                        # }
                        # runtime_details = self._client.runtimes.store(meta)
                        # runtime_uid = self._client.runtimes.get_uid(runtime_details)
                        # set the default to 3.6 runtime as 3.5 is no longer supported.
                        runtime_uid="ai-function_0.1-py3.6"
                        if not self._ICP:
                            check_runtime = requests.get(self._href_definitions.get_runtime_href(runtime_uid),  headers=self._client._get_headers())
                        else:
                            check_runtime = requests.get(self._href_definitions.get_runtime_href(runtime_uid),  headers=self._client._get_headers(),verify=False)

                        if check_runtime.status_code != 200:
                            print('No matching default runtime found. Creating one...', end="")
                            meta = {
                                self._client.runtimes.ConfigurationMetaNames.NAME: meta_props[
                                                                                       self.ConfigurationMetaNames.NAME] + "-"+"3.6",
                                self._client.runtimes.ConfigurationMetaNames.PLATFORM: {
                                    "name": "python",
                                    "version": "3.6"
                                }
                            }

                            runtime_details = self._client.runtimes.store(meta)
                            runtime_uid = self._client.runtimes.get_uid(runtime_details)
                            print('SUCCESS\n\nSuccessfully created runtime with uid: {}'.format(runtime_uid))
                        else:
                            print('Using default runtime with uid: {}'.format(runtime_uid))
                        meta_props[self.ConfigurationMetaNames.RUNTIME_UID] = runtime_uid

                function_metadata = self.ConfigurationMetaNames._generate_resource_metadata(meta_props, with_validation=True,client=self._client)
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    # For Cloud convergence, FunctionNewMetaNames is used which has path defined
                    # So, _generate_resource_metadata above will properly fill in the relevant fields
                    input_schemas = []
                    output_schemas = []
                    if self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS in meta_props and \
                            meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS] is not None:
                        input_schemas = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS]
                        function_metadata.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMAS)
                    if self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS in meta_props and \
                            meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS] is not None:
                        output_schemas = meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS]
                        function_metadata.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMAS)
                    if len(input_schemas) != 0 or len(output_schemas) != 0:
                        function_metadata.update({"schemas": {
                            "input": input_schemas,
                            "output": output_schemas}
                        })

                if self._client.CAMS and not self._client.ICP_PLATFORM_SPACES:
                    if self._client.default_space_id is not None:
                        function_metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
                    elif self._client.default_project_id is not None:
                        function_metadata['project'] = {'href': "/v2/projects/" + self._client.default_project_id}
                    else:
                        raise WMLClientError(
                            "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    if self._client.default_space_id is not None:
                        function_metadata['space_id'] = self._client.default_space_id
                    elif self._client.default_project_id is not None:
                        function_metadata['project_id'] = self._client.default_project_id
                    else:
                        raise WMLClientError(
                            "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/"
                            "client.set.default_project(<PROJECT_UID>) to proceed.")

                if not self._ICP:
                    if self._client.CLOUD_PLATFORM_SPACES:
                        response_post = requests.post(self._href_definitions.get_functions_href(),
                                                      json=function_metadata,
                                                      params=self._client._params(skip_for_create=True),
                                                      headers=self._client._get_headers())
                    else:
                        response_post = requests.post(self._href_definitions.get_functions_href(),
                                                      json=function_metadata,
                                                      headers=self._client._get_headers())
                else:
                    if self._client.ICP_PLATFORM_SPACES:
                        response_post = requests.post(self._href_definitions.get_functions_href(),
                                                      json=function_metadata,
                                                      params=self._client._params(skip_for_create=True),
                                                      headers=self._client._get_headers(),
                                                      verify=False)
                    else:
                        response_post = requests.post(self._href_definitions.get_functions_href(),
                                                      json=function_metadata,
                                                      headers=self._client._get_headers(),
                                                      verify=False)

                details = self._handle_response(expected_status_code=201,
                                                operationName=u'saving function',
                                                response=response_post)
                ##TODO_V4 Take care of this since the latest swagger endpoint is not working

                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    function_content_url = self._href_definitions.get_function_href(
                        details['metadata']['id']) + "/code"
                else:
                    function_content_url = self._href_definitions.get_function_href(details['metadata']['guid']) + "/content"

                put_header = self._client._get_headers(no_content_type=True)
                with open(content_path, 'rb') as data:
                    if not self._ICP:
                        response_definition_put = requests.put(function_content_url,
                                                               data=data,
                                                               params=self._client._params(),
                                                               headers=put_header)
                    else:
                        response_definition_put = requests.put(function_content_url,
                                                               data=data,
                                                               params = self._client._params(),
                                                               headers=put_header,
                                                               verify=False)

        except Exception as e:
            raise e
        finally:
            try:
                os.remove(archive_name)
            except:
                pass

        if self._client.WSD is None:
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                if response_definition_put.status_code != 200:
                   self._delete(details['metadata']['guid'])
                self._handle_response(200, u'saving function content', response_definition_put,json_response=False)
            else:
                if response_definition_put.status_code != 201:
                    self._delete(details['metadata']['id'])
                self._handle_response(201, u'saving function content', response_definition_put,json_response=False)

        return details

    def update(self, function_uid, changes, update_function=None):
        """
        Updates existing function metadata.

        **Parameters**

        .. important::
           #. **function_uid**:  UID of function which define what should be updated\n
              **type**: str\n
           #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
              **type**: dict\n
           #. **update_function**:  path to file with archived function content or function which should be changed for specific function_uid\n.This parameters is valid only for CP4D 3.0.0.
              **type**: str or function\n

        **Example**
         >>> metadata = {
         >>> client._functions.ConfigurationMetaNames.NAME:"updated_function"
         >>> }
         >>> function_details = client._functions.update(function_uid, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.WSD:
            raise WMLClientError(u'Updating Function is not supported for IBM Watson Studio Desktop. ')
        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and update_function is not None:
            raise WMLClientError(u'Updating Function content not supported for this release')

        self._client._check_if_either_is_set()

        function_uid = str_type_conv(function_uid)
        self._validate_type(function_uid, u'function_uid', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self.get_details(function_uid)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes,
                                                                            with_validation=True)

        url = self._href_definitions.get_function_href(function_uid)
        headers = self._client._get_headers()
        if not self._ICP:
            response = requests.patch(url, json=patch_payload, params = self._client._params(),headers=headers)
        else:
            response = requests.patch(url, json=patch_payload,params = self._client._params(), headers=headers, verify=False)
        updated_details = self._handle_response(200, u'function patch', response)

        if (self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and update_function is not None:
            self._update_function_content(function_uid,  update_function)

        return updated_details

    def _update_function_content(self, function_uid, updated_function):

        function = str_type_conv(updated_function)
        import types
        Functions._validate_type(function, u'function', [STR_TYPE, types.FunctionType], True)
        if type(function) is STR_TYPE:
            content_path = function
            user_content_file = True
        else:
            try:
                import inspect
                import gzip
                import uuid
                import re
                import shutil
                code = inspect.getsource(function).split('\n')
                r = re.compile(r"^ *")
                m = r.search(code[0])
                intend = m.group(0)

                code = [line.replace(intend, '', 1) for line in code]

                args_spec = inspect.getargspec(function)

                defaults = args_spec.defaults if args_spec.defaults is not None else []
                args = args_spec.args if args_spec.args is not None else []

                if function.__name__ is 'score':
                    code = '\n'.join(code)
                    file_content = code
                elif len(args) == len(defaults):
                    for i, d in enumerate(defaults):
                        if not is_of_python_basic_type(d):
                            raise UnexpectedType(args[i], 'primitive python type', type(d))

                    new_header = 'def {}({}):'.format(
                        function.__name__,
                        ', '.join(
                            ['{}={}'.format(arg_name, json.dumps(default)) for arg_name, default in zip(args, defaults)]
                        )
                    )

                    code[0] = new_header
                    code = '\n'.join(code)

                    file_content = """
{}

score = {}()
""".format(code, function.__name__)

                else:
                    raise WMLClientError(
                        "Function passed is not \'score\' function nor generator function. Generator function should have no arguments or all arguments with primitive python default values.")

                tmp_uid = 'tmp_python_function_code_{}'.format(str(uuid.uuid4()).replace('-', '_'))
                filename = '{}.py'.format(tmp_uid)

                with open(filename, 'w') as f:
                    f.write(file_content)

                archive_name = '{}.py.gz'.format(tmp_uid)

                with open(filename, 'rb') as f_in:
                    with gzip.open(archive_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(filename)

                content_path = archive_name
            except Exception as e:
                try:
                    os.remove(filename)
                except:
                    pass
                try:
                    os.remove(archive_name)
                except:
                    pass
                raise WMLClientError('Exception during getting function code.', e)

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            function_content_url = self._href_definitions.get_function_href(function_uid) + "/code"
        else:
            function_content_url = self._href_definitions.get_function_href(function_uid) + "/content"

        put_header = self._client._get_headers(no_content_type=True)
        with open(content_path, 'rb') as data:
            if not self._ICP:
                response_definition_put = requests.put(function_content_url, data=data,
                                                       params=self._client._params(), headers=put_header)
            else:
                response_definition_put = requests.put(function_content_url, data=data,
                                                       params=self._client._params(), headers=put_header,
                                                       verify=False)
            if response_definition_put.status_code != 200 and response_definition_put.status_code != 204 and response_definition_put.status_code != 201:
                raise WMLClientError(" Unable to update function content" + response_definition_put)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def download(self, function_uid, filename='downloaded_function.gz', rev_uid=None):
        """
            Download function content from Watson Machine Learning repository to local file.

            **Parameters**

            .. important::
                #. **function_uid**:  stored function UID\n
                   **type**: str\n
                #. **filename**:  name of local file to create (optional) Example: function_content.gz\n
                   **default value**: downloaded_function.gz\n
                   **type**: str\n


            **Output**

            .. important::

               **returns**: Path to the downloaded function content\n
               **return type**: str\n

            .. note::
               If filename is not specified, the default filename is "downloaded_function.gz".\n

            **Example**

             >>> client._functions.download(function_uid, 'my_func.tar.gz')
        """
        self._client._check_if_either_is_set()

        if os.path.isfile(filename):
            raise WMLClientError(u'File with name: \'{}\' already exists.'.format(filename))
        if rev_uid is not None and self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not applicable for this release')

        artifact_uid = str_type_conv(function_uid)
        Functions._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, True)
        filename = str_type_conv(filename)
        Functions._validate_type(filename, u'filename', STR_TYPE, True)

        artifact_url = self._href_definitions.get_function_href(artifact_uid)
        if self._client.WSD:
            import urllib
            function_get_response = requests.get(self._href_definitions.get_data_asset_href(function_uid),
                                              params=self._client._params(),
                                              headers=self._client._get_headers())

            function_details = self._handle_response(200, u'get function', function_get_response)
           # function_details = self.get_details(function_uid)
            attachment_url = function_details['attachments'][0]['object_key']
            artifact_content_url = self._href_definitions.get_wsd_model_attachment_href() + \
                                   urllib.parse.quote('wml_function/' + attachment_url, safe='')
        else:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                artifact_content_url = self._href_definitions.get_function_href(artifact_uid) + '/code'
            else:
                artifact_content_url = self._href_definitions.get_function_latest_revision_content_href(artifact_uid)

        try:
            if not self._ICP and not self._client.WSD:
                if self._client.CLOUD_PLATFORM_SPACES:
                    params = self._client._params()
                    if rev_uid is not None:
                        params.update({'rev': rev_uid})

                    r = requests.get(artifact_content_url,
                                     params=params,
                                     headers=self._client._get_headers(),
                                     stream=True)
                else:
                    r = requests.get(artifact_content_url,
                                     params= self._client._params(),
                                     headers=self._client._get_headers(),
                                     stream=True)
            else:
                params = self._client._params()

                if rev_uid is not None:
                    params.update({'revision_id': rev_uid})

                r = requests.get(artifact_content_url,
                                 params=params,
                                 headers=self._client._get_headers(),
                                 stream=True,
                                 verify=False)
            if r.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading function"), r)

            downloaded_model = r.content
            self._logger.info(u'Successfully downloaded artifact with artifact_url: {}'.format(artifact_url))
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(u'Downloading function content with artifact_url: \'{}\' failed.'.format(artifact_url), e)

        try:
            with open(filename, 'wb') as f:
                f.write(downloaded_model)
            print(u'Successfully saved function content to file: \'{}\''.format(filename))
            return os.getcwd() + "/"+filename
        except IOError as e:
            raise WMLClientError(u'Saving function content with artifact_url: \'{}\' failed.'.format(filename), e)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, function_uid):
        """
            Delete a stored function.

            **Parameters**

            .. important::
                #. **function_uid**:  stored function UID\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

             >>> client._functions.delete(function_uid)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        function_uid = str_type_conv(function_uid)
        Functions._validate_type(function_uid, u'function_uid', STR_TYPE, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and \
                self._if_deployment_exist_for_asset(function_uid):
            raise WMLClientError(
                u'Cannot delete function that has existing deployments. Please delete all associated deployments and try again')

        ##TODO_V4 Uncomment this after deployments is ready
        # Delete associated deployments, so there will be no orphans
        # if not self._client.default_project_id:
        #     deployments_details = filter(lambda x: function_uid in x['entity']['asset']['href'], self._client.deployments.get_details()['resources'])
        #     for deployment_detail in deployments_details:
        #        deployment_uid = self._client.deployments.get_uid(deployment_detail)
        #        print('Deleting orphaned function deployment \'{}\'... '.format(deployment_uid), end="")
        #        delete_status = self._client.deployments.delete(deployment_uid)
        #        print(delete_status)

        if self._client.WSD:
            function_endpoint = self._href_definitions.get_model_definition_assets_href() + "/" + function_uid
        else:
            function_endpoint = self._href_definitions.get_function_href(function_uid)
        self._logger.debug(u'Deletion artifact function endpoint: {}'.format(function_endpoint))
        if not self._ICP and not self._client.WSD:
            response_delete = requests.delete(function_endpoint, params = self._client._params(),headers=self._client._get_headers())
        else:
            if not self._client.WSD and Functions._if_deployment_exist_for_asset(self, function_uid):
                raise WMLClientError(
                    u'Cannot delete function that has existing deployments. Please delete all associated deployments and try again')
            response_delete = requests.delete(function_endpoint, params = self._client._params(), headers=self._client._get_headers(), verify=False)
        return self._handle_response(204, u'function deletion', response_delete, False)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _delete(self, function_uid):

        ##For CP4D, check if either spce or project ID is set

        function_uid = str_type_conv(function_uid)
        Functions._validate_type(function_uid, u'function_uid', STR_TYPE, True)
        ##TODO_V4 Uncomment this after deployments is ready
        # Delete associated deployments, so there will be no orphans
        if self._client.WSD:
            function_endpoint = self._href_definitions.get_model_definition_assets_href() + "/" + function_uid
        else:
            function_endpoint = self._href_definitions.get_function_href(function_uid)
        self._logger.debug(u'Deletion artifact function endpoint: {}'.format(function_endpoint))


        if not self._ICP and not self._client.WSD:
            if self._client.CLOUD_PLATFORM_SPACES and \
                    self._if_deployment_exist_for_asset(function_uid):
                raise WMLClientError(
                    u'Cannot delete function that has existing deployments. Please delete all associated deployments and try again')

            response_delete = requests.delete(function_endpoint, params=self._client._params(),
                                              headers=self._client._get_headers())
        else:
            if not self._client.WSD and self._client.ICP_30 and self._if_deployment_exist_for_asset(function_uid):
                raise WMLClientError(
                    u'Cannot delete function that has existing deployments. Please delete all associated deployments and try again')

            response_delete = requests.delete(function_endpoint, params=self._client._params(),
                                              headers=self._client._get_headers(), verify=False)
        response_delete

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, function_uid=None, limit=None):
        """
            Get metadata of function(s). If no function UID is specified all functions metadata is returned.

           **Parameters**

           .. important::
                #. **function_uid**:  UID of function (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: function(s) metadata\n
                **return type**: dict\n
                dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

           .. note::
                If UID is not specified, all functions metadata is fetched\n

           **Example**

             >>> function_details = client._functions.get_details(function_uid)
             >>> function_details = client._functions.get_details()
         """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        function_uid = str_type_conv(function_uid)
        Functions._validate_type(function_uid, u'function_uid', STR_TYPE, False)
        Functions._validate_type(limit, u'limit', int, False)
        if self._client.WSD:
            url = self._href_definitions.get_model_definition_assets_href()
            response = self._get_artifact_details(url, function_uid, limit, 'function')
            return Functions._wsd_get_required_element_from_response(self, response)
        else:
            url = self._href_definitions.get_functions_href()

        return self._get_artifact_details(url, function_uid, limit, 'functions')

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(function_details):
        """
                Get ID of stored function.

                **Parameters**

                .. important::

                   #. **function_details**:  Metadata of the stored function\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: ID of stored function\n
                    **return type**: str\n

                **Example**

                 >>> function_details = client.repository.get_function_details(function_uid)
                 >>> function_id = client._functions.get_id(function_details)
        """

        return Functions.get_uid(function_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(function_details):
        """
                Get UID of stored function. Deprecate!! Use get_id(function_details) instead

                **Parameters**

                .. important::

                   #. **function_details**:  Metadata of the stored function\n
                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: UID of stored function\n
                    **return type**: str\n

                **Example**

                 >>> function_details = client.repository.get_function_detailsf(function_uid)
                 >>> function_uid = client._functions.get_uid(function_details)
        """
        Functions._validate_type(function_details, u'function_details', object, True)
        if 'asset_id' in function_details['metadata']:
            return WMLResource._get_required_element_from_dict(function_details, u'function_details',
                                                         [u'metadata', u'asset_id'])
        else:
            if 'guid' in function_details[u'metadata']:
                Functions._validate_type_of_details(function_details, FUNCTION_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(function_details, u'function_details',
                                                                   [u'metadata', u'guid'])
            else:
                return WMLResource._get_required_element_from_dict(function_details, u'function_details',
                                                                   [u'metadata', u'id'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(function_details):
        """
            Get url of stored function.

           **Parameters**

           .. important::
                #. **function_details**:  stored function details\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of stored function\n
                **return type**: str\n

           **Example**

             >>> function_details = client.repository.get_function_detailsf(function_uid)
             >>> function_url = client._functions.get_href(function_details)
        """
        Functions._validate_type(function_details, u'function_details', object, True)
        if 'asset_type' in function_details['metadata']:
            return WMLResource._get_required_element_from_dict(function_details, u'function_details',
                                                               [u'metadata', u'href'])

            #raise WMLClientError(u'This method is not supported for IBM Watson Studio Desktop. ')
        else:
            if 'href' in function_details[u'metadata']:
                Functions._validate_type_of_details(function_details, FUNCTION_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(function_details, u'function_details', [u'metadata', u'href'])
            else:
                # Cloud Convergence
                return "/ml/v4/functions/{}".format(function_details[u'metadata'][u'id'])

    def list(self, limit=None):
        """
            List stored functions. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all functions in a table format.\n
                **return type**: None\n

           **Example**

            >>> client._functions.list()
        """
        ##For CP4D, check if either spce or project ID is set

        if self._client.WSD:
            Functions._wsd_list_assets(self, "wml_function", limit)
        else:
            self._client._check_if_either_is_set()
            function_resources = self.get_details()[u'resources']

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                function_values = [(m[u'metadata'][u'id'],
                                    m[u'metadata'][u'name'],
                                    m[u'metadata'][u'created_at'],
                                    m[u'entity'][u'type']) if 'type' in m[u'entity'] else
                                   (m[u'metadata'][u'id'],
                                    m[u'metadata'][u'name'],
                                    m[u'metadata'][u'created_at'],
                                    None) for m in function_resources]
            else:
                function_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'],
                                      m[u'entity'][u'type']) for m in function_resources]

            self._list(function_values, [u'GUID', u'NAME', u'CREATED', u'TYPE'], limit, _DEFAULT_LIST_LENGTH)

    def clone(self, function_uid, space_id=None, action="copy", rev_id=None):
        """
                Creates a new function identical with the given function either in the same space or in a new space. All dependent assets will be cloned too.

                **Parameters**

                .. important::
                    #. **model_id**:  Guid of the function to be cloned:\n

                       **type**: str\n

                    #. **space_id**: Guid of the space to which the function needs to be cloned. (optional)

                       **type**: str\n

                    #. **action**: Action specifying "copy" or "move". (optional)

                       **type**: str\n

                    #. **rev_id**: Revision ID of the function. (optional)

                       **type**: str\n

                **Output**

                .. important::

                        **returns**: Metadata of the function cloned.\n
                        **return type**: dict\n

                **Example**
                 >>> client._functions.clone(function_uid=artifact_id,space_id=space_uid,action="copy")

                .. note::
                    * If revision id is not specified, all revisions of the artifact are cloned\n

                    * Default value of the parameter action is copy\n

                    * Space guid is mandatory for move action\n

        """
        # if self._client.WSD or self._client.ICP_PLATFORM_SPACES or :
        #     raise WMLClientError(u'Clone method is not supported for IBM Watson Studio Desktop. ')

        raise WMLClientError("Not supported")

        artifact = str_type_conv(function_uid)
        Functions._validate_type(artifact, 'function_uid', STR_TYPE, True)
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

        url = self._href_definitions.get_function_href(function_uid)
        if not self._ICP:
            response_post = requests.post(url, json=clone_meta,
                                              headers=self._client._get_headers())
        else:
            response_post = requests.post(url, json=clone_meta,
                                              headers=self._client._get_headers(), verify=False)

        details = self._handle_response(expected_status_code=200, operationName=u'cloning function',
                                            response=response_post)

        return details

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_revision(self, function_uid):
        """
            Create a new functions revision.

            :param function_uid: Unique function ID
            :type function_uid: {str_type}

            Example:

            >>> client._functions.create_revision(pipeline_uid)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        function_uid = str_type_conv(function_uid)
        Functions._validate_type(function_uid, u'pipeline_uid', STR_TYPE, False)

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revision support is not there in this release')
        else:
            url = self._href_definitions.get_functions_href()
            return self._create_revision_artifact(url, function_uid, 'functions')

    def get_revision_details(self, function_uid, rev_uid):
        """
           Get metadata of specific revision of stored functions

           :param function_uid: stored functions, definition
           :type function_uid: {str_type}

           :param rev_uid: Unique id of the function revision.
           :type rev_id : int

           :returns: stored function revision metadata
           :rtype: dict

           Example:

           >>> function_revision_details = client._functions.get_details(function_uid, rev_uid)
        """
        self._client._check_if_either_is_set()
        model_uid = str_type_conv(function_uid)
        Functions._validate_type(function_uid, u'function_uid', STR_TYPE, True)
        Functions._validate_type(rev_uid, u'rev_uid', int, True)

        if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Revision APIs are not supported in this release. ')
        else:
            url = self._href_definitions.get_function_href(function_uid)
            return self._get_with_or_without_limit(url, limit=None, op_name="function",
                                                   summary=None, pre_defined=None, revision=rev_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def list_revisions(self, function_uid, limit=None):
        """
           List all revision for the given function uid.

           :param function_uid: Unique id of stored function.
           :type function_uid: {str_type}

           :param limit: limit number of fetched records (optional)
           :type limit: int

           :returns:   list all function revisions details.
           :rtype: table

           >>> details = client._functions.list_revisions(function_uid)
        """
        self._client._check_if_either_is_set()
        function_uid = str_type_conv(function_uid)

        Functions._validate_type(function_uid, u'function_uid', STR_TYPE, True)

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Revision support is not there in this release.')
        else:
            url = self._href_definitions.get_function_href(function_uid)

            # CP4D logic is wrong. By passing "revisions" in second param above for _get_artifact_details()
            # it won't even consider limit value and also GUID gives only rev number, not actual guid
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                function_resources = self._get_artifact_details(url, "revisions", limit, 'function revisions')[u'resources']
                function_values = [
                    (m[u'metadata'][u'rev'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                    function_resources]

                self._list(function_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
            else:
                function_resources = self._get_artifact_details(url + '/revisions',
                                                                None,
                                                                limit,
                                                                'function revisions')[u'resources']

                function_values = [
                    (m[u'metadata'][u'id'],
                     m[u'metadata'][u'rev'],
                     m[u'metadata'][u'name'],
                     m[u'metadata'][u'created_at']) for m in
                    function_resources]

                self._list(function_values, [u'GUID', u'rev', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)




