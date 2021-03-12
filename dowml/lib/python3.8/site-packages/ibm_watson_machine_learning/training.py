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
import json
import re
from ibm_watson_machine_learning.utils import print_text_header_h1, print_text_header_h2, TRAINING_RUN_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, str_type_conv, meta_props_str_conv, group_metrics, StatusLogger
import time
from ibm_watson_machine_learning.metanames import TrainingConfigurationMetaNames, TrainingConfigurationMetaNamesCp4d30
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_boto3.exceptions import Boto3Error


class Training(WMLResource):
    """
       Train new models.
    """

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP
        self.ConfigurationMetaNames = TrainingConfigurationMetaNames()
        if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            self.ConfigurationMetaNames = TrainingConfigurationMetaNamesCp4d30()

    @staticmethod
    def _is_training_uid(s):
        res = re.match('p\-[a-zA-Z0-9\-\_]+', s)
        return res is not None

    @staticmethod
    def _is_training_url(s):
        res = re.match('\/v3\/models\/p\-[a-zA-Z0-9\-\_]+', s)
        return res is not None

    def _is_model_definition_url(self, s):
        res = re.match('\/v2\/assets\/p\-[a-zA-Z0-9\-\_]+', s)
        return res is not None

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_status(self, training_uid):
        """
                Get the status of a training created.

                **Parameters**

                .. important::
                    #. **training_uid**:  training UID\n
                       **type**: str\n

                **Output**

                .. important::
                    **returns**: training_status\n
                    **return type**: dict\n

                **Example**

                 >>> training_status = client.training.get_status(training_uid)
        """

        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, 'training_uid', STR_TYPE, True)

        details = self.get_details(training_uid)

        if details is not None:
            return WMLResource._get_required_element_from_dict(details, u'details', [u'entity', u'status'])
        else:
            raise WMLClientError(u'Getting trained model status failed. Unable to get model details for training_uid: \'{}\'.'.format(training_uid))

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, training_uid=None, limit=None):
        """
        Get metadata of training(s). If training_uid is not specified returns all model spaces metadata.

        **Parameters**

        .. important::
            #. **training_uid**: Unique Id of Training (optional)\n
               **type**: str\n
            #. **limit**:  limit number of fetched records (optional)\n
               **type**: int\n

        **Output**

        .. important::
            **returns**: metadata of training(s)\n
            **return type**: dict\n
            The output can be {"resources": [dict]} or a dict\n

        .. note::
            If training_uid is not specified, all trainings metadata is fetched\n

        **Example**

         >>> training_run_details = client.training.get_details(training_uid)
         >>> training_runs_details = client.training.get_details()
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, 'training_uid', STR_TYPE, False)

        url = self._href_definitions.get_trainings_href()

        return self._get_artifact_details(url, training_uid, limit, 'trained models')

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(training_details):
        """
            Get training_href from training details.

            **Parameters**

            .. important::
                #. **training_details**:  Metadata of the training created\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: training href\n
                **return type**: str

            **Example**

             >>> training_details = client.training.get_details(training_uid)
             >>> run_url = client.training.get_href(training_details)
        """

        Training._validate_type(training_details, u'training_details', object, True)
        if 'id' in training_details.get('metadata'):
            training_id = WMLResource._get_required_element_from_dict(training_details, u'training_details',
                                                                      [u'metadata', u'id'])
            return "/ml/v4/trainings/"+training_id
        else:
            Training._validate_type_of_details(training_details, TRAINING_RUN_DETAILS_TYPE)
            return WMLResource._get_required_element_from_dict(training_details, u'training_details', [u'metadata', u'href'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(training_details):
        """
            Get training_uid from training details.

            **Parameters**

            .. important::
                #. **training_details**:  Metadata of the training created\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Unique id of training\n
                **return type**: str

            **Example**

             >>> training_details = client.training.get_details(training_uid)
             >>> training_uid = client.training.get_uid(training_details)

        """

        Training._validate_type(training_details, u'training_details', object, True)
        return WMLResource._get_required_element_from_dict(training_details, u'training_details', [u'metadata', u'guid'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(training_details):
        """
            Get training_id from training details.

            **Parameters**

            .. important::
                #. **training_details**:  Metadata of the training created\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Unique id of training\n
                **return type**: str

            **Example**

             >>> training_details = client.training.get_details(training_id)
             >>> training_id = client.training.get_id(training_details)

        """

        Training._validate_type(training_details, u'training_details', object, True)
        return WMLResource._get_required_element_from_dict(training_details, u'training_details',
                                                           [u'metadata', u'id'])

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def run(self, meta_props, asynchronous=True):
        """
        Create a new Machine Learning training.

        **Parameters**

        .. important::

            #. **meta_props**:  meta data of the training configuration. To see available meta names use:\n
               >>> client.training.ConfigurationMetaNames.show()

               **type**: str\n

            #. **asynchronous**:\n
               * True  - training job is submitted and progress can be checked later.\n
               * False - method will wait till job completion and print training stats.\n
               **type**: bool\n


        **Output**

        .. important::

            **returns**: Metadata of the training created\n
            **return type**: dict\n

        **Examples**
         Example meta_props for Training run creation in IBM Cloud Pak® for Data for Data version 3.0.1 or above:\n
         >>> metadata = {
         >>>  client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
         >>>  client.training.ConfigurationMetaNames.DESCRIPTION: 'Hand-written Digit Recognition Training',
         >>>  client.training.ConfigurationMetaNames.PIPELINE: {
         >>>                "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
         >>>                "rev": "12",
         >>>                "model_type": "string",
         >>>                "data_bindings": [
         >>>                  {
         >>>                    "data_reference_name": "string",
         >>>                    "node_id": "string"
         >>>                  }
         >>>                ],
         >>>                "nodes_parameters": [
         >>>                  {
         >>>                    "node_id": "string",
         >>>                    "parameters": {}
         >>>                  }
         >>>                ],
         >>>                "hardware_spec": {
         >>>                  "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
         >>>                  "rev": "12",
         >>>                  "name": "string",
         >>>                  "num_nodes": "2"
         >>>                }
         >>>      },
         >>>  client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
         >>>          'type': 's3',
         >>>          'connection': {},
         >>>          'location': {
         >>>              'href': 'v2/assets/asset1233456',
         >>>          }
         >>>          "schema": "{ \"id\": \"t1\", \"name\": \"Tasks\", \"fields\": [ { \"name\": \"duration\", \"type\": \"number\" } ]}"
         >>>      }],
         >>> client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
         >>>          'id' : 'string',
         >>>          'connection': {
         >>>              'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
         >>>              'access_key_id': '***',
         >>>              'secret_access_key': '***'
         >>>          },
         >>>          'location': {
         >>>              'bucket': 'wml-dev-results',
         >>>               'path' : "path"
         >>>          }
         >>>          'type': 's3'
         >>>      }
         >>>   }

         NOTE:  You can provide either one of the below values can be provided for training:\n
         * client.training.ConfigurationMetaNames.EXPERIMENT\n
         * client.training.ConfigurationMetaNames.PIPELINE\n
         * client.training.ConfigurationMetaNames.MODEL_DEFINITION\n
         Example meta_prop values for  training run creation in other versions:

         >>> metadata = {
         >>>  client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
         >>>  client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
         >>>          'connection': {
         >>>              'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
         >>>              'access_key_id': '***',
         >>>              'secret_access_key': '***'
         >>>          },
         >>>          'source': {
         >>>              'bucket': 'wml-dev',
         >>>          }
         >>>          'type': 's3'
         >>>      }],
         >>> client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
         >>>          'connection': {
         >>>              'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
         >>>              'access_key_id': '***',
         >>>              'secret_access_key': '***'
         >>>          },
         >>>          'target': {
         >>>              'bucket': 'wml-dev-results',
         >>>          }
         >>>          'type': 's3'
         >>>      },
         >>> client.training.ConfigurationMetaNames.PIPELINE_UID : "/v4/pipelines/<PIPELINE-ID>"
         >>> }
         >>> training_details = client.training.run(definition_uid, meta_props=metadata)
         >>> training_uid = client.training.get_uid(training_details)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(meta_props, 'meta_props', object, True)
        Training._validate_type(asynchronous, 'asynchronous', bool, True)

        meta_props_str_conv(meta_props)
        self.ConfigurationMetaNames._validate(meta_props)
        training_configuration_metadata = {
            u'training_data_references': meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES],
            u'results_reference': meta_props[self.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE]
        }
        if self.ConfigurationMetaNames.TAGS in meta_props:
            training_configuration_metadata["tags"] = meta_props[self.ConfigurationMetaNames.TAGS]

        # TODO remove when training service starts copying such data on their own

        if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            training_configuration_metadata["name"] = meta_props[self.ConfigurationMetaNames.NAME]
            training_configuration_metadata["description"] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

            if self.ConfigurationMetaNames.PIPELINE in meta_props:
                training_configuration_metadata["pipeline"] = meta_props[self.ConfigurationMetaNames.PIPELINE]
            if self.ConfigurationMetaNames.EXPERIMENT in meta_props:
                training_configuration_metadata['experiment'] = meta_props[self.ConfigurationMetaNames.EXPERIMENT]
            if self.ConfigurationMetaNames.MODEL_DEFINITION in meta_props:
                training_configuration_metadata['model_definition'] = \
                    meta_props[self.ConfigurationMetaNames.MODEL_DEFINITION]
            if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                training_configuration_metadata["space_id"] = meta_props[self.ConfigurationMetaNames.SPACE_UID]

            if self._client.default_space_id is None and self._client.default_project_id is None:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")
            else:
                if self._client.default_space_id is not None:
                    training_configuration_metadata['space_id'] = self._client.default_space_id
                elif self._client.default_project_id is not None:
                    training_configuration_metadata['project_id'] = self._client.default_project_id

            if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                if self.ConfigurationMetaNames.FEDERATED_LEARNING in meta_props:
                    training_configuration_metadata['federated_learning'] = \
                        meta_props[self.ConfigurationMetaNames.FEDERATED_LEARNING]
        else:
            if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                training_configuration_metadata["pipeline"] = {
                    "href": "/v4/pipelines/"+meta_props[self.ConfigurationMetaNames.PIPELINE_UID]
                }
                if self.ConfigurationMetaNames.PIPELINE_DATA_BINDINGS in meta_props:
                    training_configuration_metadata["pipeline"]["data_bindings"] = meta_props[self.ConfigurationMetaNames.PIPELINE_DATA_BINDINGS]
                if self.ConfigurationMetaNames.PIPELINE_NODE_PARAMETERS in meta_props:
                    training_configuration_metadata["pipeline"]["nodes_parameters"] = meta_props[
                        self.ConfigurationMetaNames.PIPELINE_NODE_PARAMETERS]
                if self.ConfigurationMetaNames.PIPELINE_MODEL_TYPE in meta_props:
                    training_configuration_metadata["pipeline"]["model_type"] = meta_props[
                        self.ConfigurationMetaNames.PIPELINE_MODEL_TYPE]
            if self.ConfigurationMetaNames.EXPERIMENT_UID in meta_props:
                training_configuration_metadata["experiment"] = {
                    "href": "/v4/experiments/" + meta_props[self.ConfigurationMetaNames.EXPERIMENT_UID]
                }

            if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                if self._client.CAMS:
                    type_uid = self._check_if_lib_or_def(meta_props[self.ConfigurationMetaNames.TRAINING_LIB_UID])
                    training_configuration_metadata["training_lib"] = {"href" : type_uid}
                else:
                    training_configuration_metadata["training_lib"]["href"] = {"href" : "/v4/libraries/" + meta_props[self.ConfigurationMetaNames.TRAINING_LIB_UID]}

                if self.ConfigurationMetaNames.COMMAND not in meta_props or self.ConfigurationMetaNames.TRAINING_LIB_RUNTIME_UID not in meta_props:
                    raise WMLClientError(u'Invalid input. command, runtime are mandatory parameter for training_lib')
                training_configuration_metadata["training_lib"].update({"command":meta_props[self.ConfigurationMetaNames.COMMAND]})
                training_configuration_metadata["training_lib"].update({"runtime": {"href" : "/v4/runtimes/"+meta_props[self.ConfigurationMetaNames.TRAINING_LIB_RUNTIME_UID]}})
                if self.ConfigurationMetaNames.TRAINING_LIB_MODEL_TYPE in meta_props:
                    training_configuration_metadata["training_lib"].update({"model_type":  meta_props[self.ConfigurationMetaNames.TRAINING_LIB_MODEL_TYPE]})
                if self.ConfigurationMetaNames.COMPUTE in meta_props:
                    training_configuration_metadata["training_lib"].update({"compute": meta_props[self.ConfigurationMetaNames.COMPUTE]})
                if self.ConfigurationMetaNames.TRAINING_LIB_PARAMETERS in meta_props:
                    training_configuration_metadata["training_lib"].update({"parameters": meta_props[self.ConfigurationMetaNames.TRAINING_LIB_PARAMETERS]})

            if self.ConfigurationMetaNames.TRAINING_LIB in meta_props:
                training_configuration_metadata["training_lib"] =  meta_props[self.ConfigurationMetaNames.TRAINING_LIB]
                # for model_definition asset - command, href and runtime are mandatory
                if self._is_model_definition_url(meta_props[self.ConfigurationMetaNames.TRAINING_LIB]['href']) is False:
                    if ('command' not in meta_props[self.ConfigurationMetaNames.TRAINING_LIB].keys() or
                            'runtime' not in meta_props[self.ConfigurationMetaNames.TRAINING_LIB].keys()):
                        raise WMLClientError(u'Invalid input. command, href, runtime are mandatory parameter for training_lib')

            if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                training_configuration_metadata["space"] = {
                    "href": "/v4/spaces/"+meta_props[self.ConfigurationMetaNames.SPACE_UID]
                }
            if self._client.CAMS:
                if self._client.default_space_id is not None:
                    training_configuration_metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
                elif self._client.default_project_id is not None:
                    training_configuration_metadata['project'] = {'href': "/v2/projects/" + self._client.default_project_id}
                else:
                    raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        train_endpoint = self._href_definitions.get_trainings_href()
        if not self._ICP:
            if self._client.CLOUD_PLATFORM_SPACES:
                params = self._client._params()
                if 'space_id' in params.keys():
                    params.pop('space_id')
                if 'project_id' in params.keys():
                    params.pop('project_id')
                response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                    params= params,
                                                    headers=self._client._get_headers())
            else:
             response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                headers=self._client._get_headers())

        else:
            if self._client.ICP_PLATFORM_SPACES:
                params = self._client._params()
                if 'space_id' in params.keys():
                    params.pop('space_id')
                if 'project_id' in params.keys():
                    params.pop('project_id')
                response_train_post = requests.post(train_endpoint,
                                                    json=training_configuration_metadata,
                                                    params= params,
                                                    headers=self._client._get_headers(),
                                                    verify=False)

            elif self._client.ICP_30:
                response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                    headers=self._client._get_headers(),verify=False)
            else:
                response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                    headers=self._client._get_headers(), verify=False)

        run_details = self._handle_response(201, u'training', response_train_post)

        trained_model_guid = self.get_uid(run_details)

        if asynchronous is True:
            return run_details
        else:
            print_text_header_h1(u'Running \'{}\''.format(trained_model_guid))

            status = self.get_status(trained_model_guid)
            state = status[u'state']

            with StatusLogger(state) as status_logger:
                while state not in ['error', 'completed', 'canceled']:
                    time.sleep(5)
                    state = self.get_status(trained_model_guid)['state']
                    status_logger.log_state(state)

            if u'completed' in state:
                print(u'\nTraining of \'{}\' finished successfully.'.format(str(trained_model_guid)))
            else:
                print(u'\nTraining of \'{}\' failed with status: \'{}\'.'.format(trained_model_guid, str(status)))

            self._logger.debug(u'Response({}): {}'.format(state, run_details))
            return self.get_details(trained_model_guid)

    def list(self, limit=None):
        """
           List stored trainings. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all trainings in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.training.list()
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        details = self.get_details(limit=limit)
        resources = details[u'resources']
        values = [(m[u'metadata'][u'guid'], m[u'entity'][u'status'][u'state'], m[u'metadata'][u'created_at']) for m in resources]

        self._list(values, [u'GUID (training)', u'STATE', u'CREATED'], limit, 50)

    def list_subtrainings(self, training_uid):
        """
           List the sub-trainings.

           **Parameters**

           .. important::
                #. **training_uid**:  Training GUID\n
                   **type**: str\n

           **Output**

           .. important::
                This method only prints the list of all sub-trainings associated with a training in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.training.list_subtrainings()

        """
        ##For CP4D, check if either spce or project ID is set
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Sub-trainings are no longer available for trainings, please use list_intermediate_models().")

        self._client._check_if_either_is_set()
        details = self.get_details(training_uid)
        verify=True
        if self._client.ICP:
            verify=False
        if "experiment" not in details["entity"]:
            raise WMLClientError("Sub-trainings are available for training created via experiment only.")
        details_parent = requests.get(
            self._wml_credentials['url'] + '/v4/trainings?parent_id=' + training_uid,
            params=self._client._params(),
            headers=self._client._get_headers(),
            verify=verify
        )
        details_json = self._handle_response(200, "Get training details", details_parent)
        resources = details_json["resources"]
        values = [(m[u'metadata'][u'guid'], m[u'entity'][u'status'][u'state'], m[u'metadata'][u'created_at']) for m in resources]

        self._list(values, [u'GUID (sub_training)', u'STATE', u'CREATED'], None, 50)

    def list_intermediate_models(self, training_uid):
        """
           List the intermediate_models.

           **Parameters**

           .. important::
                #. **training_uid**:  Training GUID\n
                   **type**: str\n

           **Output**

           .. important::
                This method only prints the list of all intermediate_models associated with an AUTOAI training in a table format.\n
                **return type**: None\n

           .. note::

                This method prints the training logs.
                This method is not supported for IBM Cloud Pak® for Data.

           **Example**

            >>> client.training.list_intermediate_models()

        """
        ##For CP4D, check if either spce or project ID is set
        if self._client.ICP_30 or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("This method is not supported for IBM Cloud Pak® for Data. " )

        self._client._check_if_either_is_set()
        details = self.get_details(training_uid)
        #if status is completed then only lists only global_output else display message saying "state value"
        training_state = details[u'entity'][u'status'][u'state']
        if(training_state=='completed'):

            if 'metrics' in details[u'entity'][u'status'] and details[u'entity'][u'status'].get('metrics') is not None:
                metrics_list = details[u'entity'][u'status'][u'metrics']
                new_list=[]
                for ml in metrics_list:
                    # if(ml[u'context'][u'intermediate_model'][u'process']=='global_output'):
                    if 'context' in ml and 'intermediate_model' in ml[u'context']:
                        name = ml[u'context'][u'intermediate_model'].get('name', "")
                        if 'location' in ml[u'context'][u'intermediate_model']:
                            path = ml[u'context'][u'intermediate_model'][u'location'].get('model', "")
                        else:
                            path = ""
                    else:
                        name = ""
                        path = ""

                    accuracy=ml[u'ml_metrics'].get('training_accuracy', "")
                    F1Micro=round(ml[u'ml_metrics'].get('training_f1_micro', 0), 2)
                    F1Macro = round(ml[u'ml_metrics'].get('training_f1_macro', 0), 2)
                    F1Weighted = round(ml[u'ml_metrics'].get('training_f1_weighted', 0), 2)
                    logLoss=round(ml[u'ml_metrics'].get('training_neg_log_loss', 0), 2)
                    PrecisionMicro = round(ml[u'ml_metrics'].get('training_precision_micro', 0), 2)
                    PrecisionWeighted = round(ml[u'ml_metrics'].get('training_precision_weighted', 0), 2)
                    PrecisionMacro = round(ml[u'ml_metrics'].get('training_precision_macro', 0), 2)
                    RecallMacro = round(ml[u'ml_metrics'].get('training_recall_macro', 0), 2)
                    RecallMicro = round(ml[u'ml_metrics'].get('training_recall_micro', 0), 2)
                    RecallWeighted = round(ml[u'ml_metrics'].get('training_recall_weighted', 0), 2)
                    createdAt = details[u'metadata'][u'created_at']
                    new_list.append([name,path,accuracy,F1Micro,F1Macro,F1Weighted,logLoss,PrecisionMicro,PrecisionMacro,PrecisionWeighted,RecallMicro,RecallMacro,RecallWeighted,createdAt])
                    new_list.append([])

                from tabulate import tabulate
                header = [u'NAME', u'PATH', u'Accuracy', u'F1Micro', u'F1Macro', u'F1Weighted', u'LogLoss', u'PrecisionMicro' , u'PrecisionMacro',u'PrecisionWeighted', u'RecallMicro', u'RecallMacro', u'RecallWeighted', u'CreatedAt' ]
                table = tabulate([header] + new_list)

                print(table)
                #self._list(new_list, [u'NAME', u'PATH', u'Accuracy', u'F1Micro', u'F1Macro', u'F1Weighted', u'LogLoss', u'PrecisionMicro' , u'PrecisionMacro',u'PrecisionWeighted', u'RecallMicro', u'RecallMacro', u'RecallWeighted', u'CreatedAt' ], None, 50)
            else:
                print(" There is no intermediate model metrics are available for this training uid. ")
        else:
            self._logger.debug("state is not completed")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def cancel(self, training_uid, hard_delete=False):
        """
        Cancel a training which is currently running and remove it. This method is also be used to delete metadata
        details of the completed or canceled training run when hard_delete parameter is set to True.

        **Parameters**

        .. important::
            #. **training_uid**:  Training UID\n
               **type**: str\n

            #. **hard_delete**: specify True or False.
               True - To delete the completed or canceled training runs.
               False - To cancel the currently running training run. Default value is False.
               **type**: Boolean\n

        **Output**

        .. important::
            **returns**: status ("SUCCESS" or "FAILED")\n
            **return type**: str\n

        **Example**

         >>> client.training.cancel(training_uid)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, u'training_uid', STR_TYPE, True)
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
        else:
            params = None

        if hard_delete is True:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                params.update({'hard_delete': u'true'})
            else:
                params = {}
                params.update({'hard_delete': u'true'})

        if not self._ICP and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            response_delete = requests.delete(self._href_definitions.get_training_href(training_uid),
                                          headers=self._client._get_headers(),params=params)
        else:
            response_delete = requests.delete(self._href_definitions.get_training_href(training_uid),
                                              headers=self._client._get_headers(),params=params,verify=False)

        if response_delete.status_code == 400 and \
           response_delete.text is not None and 'Job already completed with state' in response_delete.text:
            print("Job is not running currently. Please use 'hard_delete=True' parameter to force delete"
                  " completed or canceled training runs.")
            return "SUCCESS"
        else:
            return self._handle_response(204, u'trained model deletion', response_delete, False)

    def _COS_logs(self, run_uid,on_start=lambda: {}):
        on_start()
        run_details = self.get_details(run_uid)
        if 'connection' in run_details["entity"]["results_reference"] and run_details["entity"]["results_reference"].get("connection") is not None:
            endpoint_url = run_details["entity"]["results_reference"]["connection"]["endpoint_url"]
            aws_access_key = run_details["entity"]["results_reference"]["connection"]["access_key_id"]
            aws_secret = run_details["entity"]["results_reference"]["connection"]["secret_access_key"]
            bucket = run_details["entity"]["results_reference"]["location"]["bucket"]
            # try:
            #     run_details["entity"]["training_results_reference"]["location"]["model_location"]
            # except:
            #     raise WMLClientError("The training-run has not started. Error - " + run_details["entity"]["status"]["error"]["errors"][0]["message"])

            if (bucket == ""):
                bucket = run_details["entity"]["results_reference"]["target"]["bucket"]
            import ibm_boto3

            client_cos = ibm_boto3.client(service_name='s3', aws_access_key_id=aws_access_key,
                                          aws_secret_access_key=aws_secret,
                                          endpoint_url=endpoint_url)

            try:
                if self._client.CLOUD_PLATFORM_SPACES:
                    logs = run_details["entity"].get("results_reference").get("location").get("logs")
                    if logs is None:
                        print(" There is no logs details for this Training run, hence no logs.")
                        return

                    key = logs + "/learner-1/training-log.txt"

                else:
                    try:
                        key = "data/" + run_details["metadata"]["guid"] + "/pipeline-model.json"

                        obj = client_cos.get_object(Bucket=bucket, Key=key)
                        pipeline_model = json.loads((obj['Body'].read().decode('utf-8')))

                    except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:
                        if ex.response['Error']['Code'] == 'NoSuchKey':
                            print(" Error - There is no training logs are found for the given training run id")
                            return
                         #   print("ERROR - Cannot find pipeline_model.json in the bucket "+ run_uid)
                        else:
                            print(ex)
                            return
                    if pipeline_model is not None:
                        key = pipeline_model["pipelines"][0]["nodes"][0]["parameters"]["model_id"] + "/learner-1/training-log.txt"
                    else:
                        print(" Error - Cannot find the any logs for the given training run id")
                obj = client_cos.get_object(Bucket=bucket, Key=key)
                print(obj['Body'].read().decode('utf-8'))
            except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:

                if ex.response['Error']['Code'] == 'NoSuchKey':
                    print("ERROR - Cannot find training-log.txt in the bucket")
                else:
                    print(ex)
                    print("ERROR - Cannot get the training run log in the bucket")
        else:
            print(" There is no connection details for this Training run, hence no logs.")


    def _COS_metrics(self, run_uid,on_start=lambda: {}):
        on_start()
        run_details = self.get_details(run_uid)
        endpoint_url = run_details["entity"]["results_reference"]["connection"]["endpoint_url"]
        aws_access_key = run_details["entity"]["results_reference"]["connection"]["access_key_id"]
        aws_secret = run_details["entity"]["results_reference"]["connection"]["secret_access_key"]
        bucket = run_details["entity"]["results_reference"]["location"]["bucket"]
        # try:
        #     run_details["entity"]["training_results_reference"]["location"]["model_location"]
        # except:
        #     raise WMLClientError("The training-run has not started. Error - " + run_details["entity"]["status"]["error"]["errors"][0]["message"])

        if (bucket == ""):
            bucket = run_details["entity"]["results_reference"]["target"]["bucket"]
        import ibm_boto3

        client_cos = ibm_boto3.client(service_name='s3', aws_access_key_id=aws_access_key,
                                      aws_secret_access_key=aws_secret,
                                      endpoint_url=endpoint_url)

        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                logs = run_details["entity"].get("results_reference").get("location").get("logs")
                if logs is None:
                    print(" Metric log location details for this Training run is not available.")
                    return
                key = logs + "/learner-1/evaluation-metrics.txt"
            else:
                try:
                    key = run_details["metadata"]["guid"] + "/pipeline-model.json"

                    obj = client_cos.get_object(Bucket=bucket, Key=key)

                    pipeline_model = json.loads((obj['Body'].read().decode('utf-8')))
                except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:

                    if ex.response['Error']['Code'] == 'NoSuchKey':
                        print("ERROR - Cannot find pipeline_model.json in the bucket for training id "+ run_uid)
                        print("There is no training logs are found for the given training run id")
                        return
                    else:
                        print(ex)
                        return
                key = pipeline_model["pipelines"][0]["nodes"][0]["parameters"].get["model_id"] + "/learner-1/evaluation-metrics.txt"

            obj = client_cos.get_object(Bucket=bucket, Key=key)
            print(obj['Body'].read().decode('utf-8'))

        except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                print("ERROR - Cannot find evaluation-metrics.txt in the bucket")
            else:
                print(ex)
                print("ERROR - Cannot get the location of evaluation-metrics.txt details in the bucket")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def monitor_logs(self, training_uid):
        """
            Monitor the logs of a training created.

            **Parameters**

            .. important::
                #. **training_uid**:  Training UID\n
                   **type**: str\n

            **Output**

            .. important::

                **returns**: None\n
                **return type**: None\n

            .. note::

                This method prints the training logs.
                This method is not supported for IBM Cloud Pak® for Data.

            **Example**

             >>> client.training.monitor_logs(training_uid)

        """

        if self._client.ICP_30 or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Metrics logs are not supported. This method is not supported for IBM Cloud Pak® for Data. ")

        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, u'training_uid', STR_TYPE, True)

        self._simple_monitor_logs(training_uid, lambda: print_text_header_h1(u'Log monitor started for training run: ' + str(training_uid)))

        print_text_header_h2('Log monitor done.')

    def _simple_monitor_logs(self, training_uid, on_start=lambda: {}):
        try:
            run_details = self.get_details(training_uid)
        except ApiRequestFailure as ex:
            if "404" in str(ex.args[1]):
                print("Could not find the training run details for the given training run id. ")
                return
            else:
                raise ex

        status = run_details["entity"]["status"]["state"]

        if (status == "completed" or status == "error" or status == "failed" or status == "canceled"):
            self._COS_logs(training_uid,
                           lambda: print_text_header_h1(u'Log monitor started for training run: ' + str(training_uid)))
        else:
            from lomond import WebSocket
            if not self._ICP:
                if self._client.CLOUD_PLATFORM_SPACES:
                    ws_param = self._client._params()
                    if 'project_id' in ws_param.keys():
                        proj_id = ws_param.get('project_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?project_id=" + proj_id
                    else:
                        space_id = ws_param.get('space_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?space_id=" + space_id
                else:
                    monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                     u'wss') + u'/v4/trainings/' + training_uid
            else:
                monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                         u'wss') + u'/v4/trainings/' + training_uid
            websocket = WebSocket(monitor_endpoint)
            try:
                websocket.add_header(bytes("Authorization", "utf-8"), bytes("Bearer " + self._client.service_instance._get_token(), "utf-8"))
            except:
                websocket.add_header(bytes("Authorization"), bytes("bearer " + self._client.service_instance._get_token()))

            on_start()

            for event in websocket:

                if event.name == u'text':
                    text = json.loads(event.text)
                    entity = text[u'entity']
                    if 'status' in entity:
                      if 'message' in entity['status']:
                        message = entity['status']['message']
                        if len(message) > 0:
                          print(message)

            websocket.close()

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def monitor_metrics(self, training_uid):
        """
            Monitor the metrics of a training created.

            **Parameters**

            .. important::
                #. **training_uid**:  Training UID\n
                   **type**: str\n

            **Output**

            .. important::

                **returns**: None\n
                **return type**: None\n

            .. note::

                This method prints the training metrics.
                This method is not supported for IBM Cloud Pak® for Data.

            **Example**

             >>> client.training.monitor_metrics(training_uid)
        """
        if self._client.ICP_30 or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Metrics monitoring is not supported for IBM Cloud Pak® for Data")

        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, u'training_uid', STR_TYPE, True)
        try:
            run_details = self.get_details(training_uid)
        except ApiRequestFailure as ex:
            if "404" in str(ex.args[1]):
                print("Could not find the training run details for the given training run id. ")
                return
            else:
                raise ex
        status = run_details["entity"]["status"]["state"]

        if (status == "completed" or status == "error" or status == "failed" or status == "canceled"):
            self._COS_metrics(training_uid,
                           lambda: print_text_header_h1(u'Log monitor started for training run: ' + str(training_uid)))
        else:
            from lomond import WebSocket
            if not self._ICP:
                if self._client.CLOUD_PLATFORM_SPACES:
                    ws_param = self._client._params()
                    if 'project_id' in ws_param.keys():
                        proj_id = ws_param.get('project_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?project_id=" + proj_id
                    else:
                        space_id = ws_param.get('space_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?space_id=" + space_id

                else:
                    monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                     u'wss') + u'/v4/trainings/' + training_uid
            else:
                monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                         u'wss') + u'/v4/trainings/' + training_uid
            websocket = WebSocket(monitor_endpoint)
            try:
                websocket.add_header(bytes("Authorization", "utf-8"), bytes("Bearer " + self._client.service_instance._get_token(), "utf-8"))
            except:
                websocket.add_header(bytes("Authorization"), bytes("bearer " + self._client.service_instance._get_token()))

            print_text_header_h1('Metric monitor started for training run: ' + str(training_uid))

            for event in websocket:
                if event.name == u'text':
                    text = json.loads(event.text)
                    entity = text[u'entity']
                    if 'status' in entity:
                        status = entity[u'status']
                        if u'metrics' in status:
                            metrics = status[u'metrics']
                            if len(metrics) > 0:
                             metric = metrics[0]
                             print(metric)

            websocket.close()

            print_text_header_h2('Metric monitor done.')

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_metrics(self, training_uid):
        """
                Get metrics.

                **Parameters**

                .. important::
                    #. **training_uid**:  training UID\n
                       **type**: str\n

                **Output**

                .. important::
                    **returns**: Metrics of a training run\n
                    **return type**: list of dict\n

                **Example**

                 >>> training_status = client.training.get_metrics(training_uid)

        """

        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, u'training_uid', STR_TYPE, True)
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            entity = self.get_details(training_uid)
            if 'metrics' in entity:
                metrics = entity['metrics']
                return metrics
            else:
                raise WMLClientError(" There is no Metrics details are available for the given training_uid")
        else:
            status = self.get_status(training_uid)
            if 'metrics' in status:
                metrics = status['metrics']
                return metrics
            else:
                raise WMLClientError(" There is no Metrics details are available for the given training_uid")

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _get_latest_metrics(self, training_uid):
        """
             Get latest metrics values.

             :param training_uid: ID of trained model
             :type training_uid: {0}

             :returns: metric values
             :rtype: list of dicts

             A way you might use me is:

             >>> client.training.get_latest_metrics(training_uid)
         """
        training_uid = str_type_conv(training_uid)
        Training._validate_type(training_uid, u'training_uid', STR_TYPE, True)

        status = self.get_status(training_uid)
        metrics = status.get('metrics', [])
        latest_metrics = []

        if len(metrics) > 0:
            grouped_metrics = group_metrics(metrics)

            for key, value in grouped_metrics.items():
                sorted_value = sorted(value, key=lambda k: k['iteration'])

            latest_metrics.append(sorted_value[-1])

        return latest_metrics