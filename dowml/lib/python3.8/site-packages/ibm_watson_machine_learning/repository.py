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
from ibm_watson_machine_learning.utils import get_url, INSTANCE_DETAILS_TYPE, STR_TYPE, STR_TYPE_NAME, docstring_parameter, str_type_conv, is_python_2
from ibm_watson_machine_learning.metanames import ModelMetaNames, ExperimentMetaNames, FunctionMetaNames, PipelineMetanames, SpacesMetaNames, MemberMetaNames, FunctionNewMetaNames
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.models import Models
from ibm_watson_machine_learning.experiments import Experiments
from ibm_watson_machine_learning.functions import Functions
from ibm_watson_machine_learning.pipelines import Pipelines
from ibm_watson_machine_learning.spaces import Spaces
from multiprocessing import Pool
from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryClient
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS
import os
_DEFAULT_LIST_LENGTH = 50


class Repository(WMLResource):
    """
    Store and manage your models, functions, spaces, pipelines and experiments using Watson Machine Learning Repository.
    
    .. important::
    
        #. To view ModelMetaNames, use: \n 
           >>> client.repository.ModelMetaNames.show()
        #. To view ExperimentMetaNames, use: \n 
           >>> client.repository.ExperimentMetaNames.show()
        #. To view FunctionMetaNames, use: \n 
           >>> client.repository.FunctionMetaNames.show()
        #. To view PipelineMetaNames, use: \n 
           >>> client.repository.PipelineMetaNames.show()

    """

    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.WSD and not client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            Repository._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Repository._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self._ICP = client.ICP
        self._WSD = client.WSD
        self._ml_repository_client = None
        Repository.cloud_platform_spaces = client.CLOUD_PLATFORM_SPACES
        Repository.icp_platform_spaces = client.ICP_PLATFORM_SPACES

        self.ExperimentMetaNames = ExperimentMetaNames()
        if not client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            self.FunctionMetaNames = FunctionMetaNames()
        else:
            self.FunctionMetaNames = FunctionNewMetaNames()
        self.PipelineMetaNames = PipelineMetanames()
        self.SpacesMetaNames = SpacesMetaNames()
        self.ModelMetaNames = ModelMetaNames()
        self.MemberMetaNames = MemberMetaNames()

        self._refresh_repo_client() # regular token is initialized in service_instance

    def _refresh_repo_client(self):
        # If apiKey is passed in credentials then refresh repoclient with IAM token else MLToken
        self._ml_repository_client = MLRepositoryClient(self._wml_credentials[u'url'])
        if self._client.proceed is True:
            if self._client.service_instance._is_iam() is not None:
                self._ml_repository_client.authorize_with_token(self._client.wml_token)
                self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                if self._client.project_id is not None:
                    self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)
            else:
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    platform_spaces = True
                else:
                    platform_spaces = False

                self._ml_repository_client.authorize_with_iamtoken(self._client.wml_token,
                                                                   self._wml_credentials[u'instance_id'],
                                                                   platform_spaces)
                self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                # Cloud Convergence
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                   self._ml_repository_client._add_header('ML-Instance-ID', self._wml_credentials[u'instance_id'])
                if self._client.project_id is not None:
                    self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)
        else:
            if self._client._is_IAM():
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    platform_spaces = True
                else:
                    platform_spaces = False

                self._ml_repository_client.authorize_with_iamtoken(self._client.wml_token,
                                                                   self._wml_credentials[u'instance_id'],
                                                                   platform_spaces)
                self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                # Cloud Convergence
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                   self._ml_repository_client._add_header('ML-Instance-ID', self._wml_credentials[u'instance_id'])
                if self._client.project_id is not None:
                    self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)
            else:
                if self._ICP:
                    self._repotoken = self._client._get_icptoken()
                    self._ml_repository_token = self._repotoken.replace('Bearer', '')
                    self._ml_repository_client.authorize_with_token(self._ml_repository_token)
                else:
                    if not self._client.WSD:
                        self._ml_repository_client.authorize(self._wml_credentials[u'username'], self._wml_credentials[u'password'])
                        self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                        if self._client.project_id is not None:
                           self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)

    def store_experiment(self, meta_props):
        """
        Create an experiment.

        **Parameters**

        .. important::
            #. **meta_props**:  meta data of the experiment configuration. To see available meta names use:\n
                >>> client.experiments.ConfigurationMetaNames.get()
               **type**: dict\n

        **Output**

        .. important::

            **returns**: Metadata of the experiment created\n
            **return type**: dict\n

        **Example**

        >>> metadata = {
        >>>  client.experiments.ConfigurationMetaNames.NAME: 'my_experiment',
        >>>  client.experiments.ConfigurationMetaNames.EVALUATION_METRICS: ['accuracy'],
        >>>  client.experiments.ConfigurationMetaNames.TRAINING_REFERENCES: [
        >>>      {
        >>>        'pipeline': {'href': pipeline_href_1}
        >>>      },
        >>>      {
        >>>        'pipeline': {'href':pipeline_href_2}
        >>>      },
        >>>   ]
        >>> }
        >>> experiment_details = client.repository.store_experiment(meta_props=metadata)
        >>> experiment_href = client.repository.get_experiment_href(experiment_details)
        """

        if self._client.WSD:
            raise WMLClientError(u'Experiment APIs are not supported in Watson Studio Desktop.')

        return self._client.experiments.store(meta_props)


    def store_space(self, meta_props):
        """
               Create a space.

               **Parameters**

               .. important::

                    #. **meta_props**:  meta data of the space configuration. To see available meta names use:\n
                                        >>> client.spaces.ConfigurationMetaNames.get()

                       **type**: dict\n

               **Output**

               .. important::

                    **returns**: Metadata of the space created\n
                    **return type**: dict\n

               **Example**

                >>> metadata = {
                >>>  client.spaces.ConfigurationMetaNames.NAME: 'my_space'
                >>> }
                >>> space_details = client.repository.store_space(meta_props=metadata)
                >>> space_href = client.repository.get_space_href(experiment_details)
        """


        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.store(meta_props)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_member(self, space_uid,meta_props):
        """
                Create a member within a space.

                **Parameters**

                .. important::
                   #. **meta_props**:  meta data of the member configuration. To see available meta names use:\n
                                    >>> client.spaces.ConfigurationMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of the stored member\n
                    **return type**: dict\n

                .. note::
                    * client.spaces.MemberMetaNames.ROLE can be any one of the following "viewer, editor, admin"\n
                    * client.spaces.MemberMetaNames.IDENTITY_TYPE can be any one of the following "user,service"\n
                    * client.spaces.MemberMetaNames.IDENTITY can be either service-ID or IAM-userID\n

                **Example**

                 >>> metadata = {
                 >>>  client.spaces.MemberMetaNames.ROLE:"Admin",
                 >>>  client.spaces.MemberMetaNames.IDENTITY:"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                 >>>  client.spaces.MemberMetaNames.IDENTITY_TYPE:"service"
                 >>> }
                 >>> members_details = client.repository.create_member(space_uid=space_id, meta_props=metadata)
        """
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.create_member(space_uid,meta_props)

    @staticmethod
    def _meta_props_to_repository_v3_style(meta_props):
        if is_python_2():
            new_meta_props = meta_props.copy()

            for key in new_meta_props:
                if type(new_meta_props[key]) is unicode:
                    new_meta_props[key] = str(new_meta_props[key])

            return new_meta_props
        else:
            return meta_props

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store_pipeline(self, meta_props):
        """
                Create a pipeline.

                **Parameters**

                .. important::

                   #. **meta_props**:  meta data of the pipeline configuration. To see available meta names use:\n

                        >>> client.pipelines.ConfigurationMetaNames.get()

                      **type**: dict\n

                **Output**

                .. important::

                    **returns**: Metadata of the pipeline created\\n
                    **return type**: dict\n

                **Example**

                    >>> metadata = {
                    >>>  client.pipelines.ConfigurationMetaNames.NAME: 'my_training_definition',
                    >>>  client.pipelines.ConfigurationMetaNames.DOCUMENT: {"doc_type":"pipeline","version": "2.0","primary_pipeline": "dlaas_only","pipelines": [{"id": "dlaas_only","runtime_ref": "hybrid","nodes": [{"id": "training","type": "model_node","op": "dl_train","runtime_ref": "DL","inputs": [],"outputs": [],"parameters": {"name": "tf-mnist","description": "Simple MNIST model implemented in TF","command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000","compute": {"name": "k80","nodes": 1},"training_lib_href":"/v4/libraries/64758251-bt01-4aa5-a7ay-72639e2ff4d2/content"},"target_bucket": "wml-dev-results"}]}]}}
                    >>> pipeline_details = client.repository.store_pipeline(pipeline_filepath, meta_props=metadata)
                    >>> pipeline_href = client.repository.get_pipeline_href(pipeline_details)
        """


        return self._client.pipelines.store(meta_props)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store_model(self, model, meta_props=None, training_data=None, training_target=None, pipeline=None, feature_names=None, label_column_names=None,subtrainingId=None):
        """
                Create a model.

                **Parameters**

                .. important::

                    #. **model**:  \n
                            Can be one of following:\n
                            - The train model object:\n
                                - scikit-learn
                                - xgboost
                                - spark (PipelineModel)
                            - path to saved model in format:\n
                                - keras (.tgz)
                                - pmml (.xml)
                                - scikit-learn (.tar.gz)
                                - tensorflow (.tar.gz)
                                - spss (.str)
                            - directory containing model file(s):\n
                                - scikit-learn
                                - xgboost
                                - tensorflow
                            - unique id of trained model
                    #. **training_data**:  Spark DataFrame supported for spark models. Pandas dataframe, numpy.ndarray or array supported for scikit-learn models\n
                       **type**: spark dataframe, pandas dataframe, numpy.ndarray or array\n

                    #. **meta_props**:  meta data of the models configuration. To see available meta names use:\n
                                        >>> client.repository.ModelMetaNames.get()

                       **type**: dict\n

                    #. **training_target**:  array with labels required for scikit-learn models\n
                       **type**: array\n

                    #. **pipeline**:  pipeline required for spark mllib models\n
                       **type**: object\n

                    #. **feature_names**:  Feature names for the training data in case of Scikit-Learn/XGBoost models. This is applicable only in the case where the training data is not of type - pandas.DataFrame.\n
                       **type**: numpy.ndarray or list\n

                    #. **label_column_names**:  Label column names of the trained Scikit-Learn/XGBoost models.\n
                       **type**: numpy.ndarray and list\n



                **Output**

                .. important::

                    **returns**: Metadata of the model created\n
                    **return type**: dict\n

                .. note::

                    * For a keras model, model content is expected to contain a .h5 file and an archived version of it.\n

                    * feature_names is an optional argument containing the feature names for the training data in case of Scikit-Learn/XGBoost models. Valid types are numpy.ndarray and list. This is applicable only in the case where the training data is not of type - pandas.DataFrame.\n

                    * If the training data is of type pandas.DataFrame and feature_names are provided, feature_names are ignored.\n

                    * The value can be a single dictionary(being deprecated, use list even for single schema) or
                      a list if you are using single input data schema. you can provide multiple schemas as dictionaries inside a list.
                **Example**

                    >>> stored_model_details = client.repository.store_model(model, name)

                    In more complicated cases you should create proper metadata, similar to this one:\n

                    >>> sw_spec_id = client.software_specifications.get_id_by_name('scikit-learn_0.23-py3.7')
                    >>> sw_spec_id
                    >>> metadata = {
                    >>>        client.repository.ModelMetaNames.NAME: 'customer satisfaction prediction model',
                    >>>        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
                    >>>        client.repository.ModelMetaNames.TYPE: 'scikit-learn_0.23'
                    >>>}

                    In case when you want to provide input data schema of the model, you can provide it as part of meta

                    >>> sw_spec_id = client.software_specifications.get_id_by_name('spss-modeler_18.1')
                    >>> sw_spec_id
                    >>> metadata = {
                    >>>        client.repository.ModelMetaNames.NAME: 'customer satisfaction prediction model',
                    >>>        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
                    >>>        client.repository.ModelMetaNames.TYPE: 'spss-modeler_18.1',
                    >>>        client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: [{'id': 'test',
                    >>>                                                             'type': 'list',
                    >>>                                                             'fields': [{'name': 'age', 'type': 'float'},
                    >>>                                                                        {'name': 'sex', 'type': 'float'},
                    >>>                                                                         {'name': 'fbs', 'type': 'float'},
                    >>>                                                                         {'name': 'restbp', 'type': 'float'}]
                    >>>                                                               },
                    >>>                                                               {'id': 'test2',
                    >>>                                                                'type': 'list',
                    >>>                                                                'fields': [{'name': 'age', 'type': 'float'},
                    >>>                                                                           {'name': 'sex', 'type': 'float'},
                    >>>                                                                           {'name': 'fbs', 'type': 'float'},
                    >>>                                                                           {'name': 'restbp', 'type': 'float'}]
                    >>>                                                               }]
                    >>>             }

                    A way you might use me with local tar.gz containing model:\n

                    >>> stored_model_details = client.repository.store_model(path_to_tar_gz, meta_props=metadata, training_data=None)

                    A way you might use me with local directory containing model file(s):\n

                    >>> stored_model_details = client.repository.store_model(path_to_model_directory, meta_props=metadata, training_data=None)

                    A way you might use me with trained model guid:\n

                    >>> stored_model_details = client.repository.store_model(trained_model_guid, meta_props=metadata, training_data=None)
            """

        return self._client._models.store(model, meta_props=meta_props, training_data=training_data, training_target=training_target, pipeline=pipeline, feature_names=feature_names, label_column_names=label_column_names,subtrainingId=subtrainingId)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def clone(self, artifact_id, space_id=None, action="copy", rev_id=None):
        """
            Creates a new resource(models, runtimes, libraries, experiments, functions, pipelines) identical with the model either in the same space or in a new space. All dependent assets will be cloned too.

            **Parameters**

            .. important::
                #. **model_id**:  Guid of the artifact to be cloned:\n

                   **type**: str\n

                #. **space_id**: Guid of the space to which the model needs to be cloned. (optional)

                   **type**: str\n

                #. **action**: Action specifying "copy" or "move". (optional)

                   **type**: str\n

                #. **rev_id**: Revision ID of the artifact. (optional)

                   **type**: str\n

            **Output**

            .. important::

                    **returns**: Metadata of the model cloned.\n
                    **return type**: dict\n

            **Example**

             >>> client.repository.clone(artifact_id=artifact_id,space_id=space_uid,action="copy")

            .. note::
                * If revision id is not specified, all revisions of the artifact are cloned\n

                * Default value of the parameter action is copy\n

                * Space guid is mandatory for move action\n

            """
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError('Cloning is not supported.')
        artifact = str_type_conv(artifact_id)
        Models._validate_type(artifact, 'artifact_id', STR_TYPE, True)
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
        res = self._check_artifact_type(artifact_id)

        url = ""
        type = ""
        if res['model'] is True:
            url = self._href_definitions.get_published_model_href(artifact_id)
            type = "model"
        elif res['library'] is True:
            url = self._href_definitions.get_custom_library_href(artifact_id)
            type = "library"
        elif res['runtime'] is True:
            url = self._href_definitions.get_runtime_href(artifact_id)
            type = "runtime"
        elif res['function'] is True:
            url = self._href_definitions.get_function_href(artifact_id)
            type = "function"
        elif res['pipeline'] is True:
            url = self._href_definitions.get_pipeline_href(artifact_id)
            type = "pipeline"
        elif res['experiment'] is True:
            url = self._href_definitions.get_experiment_href(artifact_id)
            type = "experiment"


        if type == "":
            raise WMLClientError('Unsupported artifact type. Supported artifact types are models, libraries, runtimes, experiments, pipelines and functions')
        if not self._ICP:
            response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers())
        else:
            response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers(), verify=False)

        details = self._handle_response(expected_status_code=200, operationName=u'cloning '+ type,
                                        response=response_post)

        return details

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def store_function(self, function, meta_props):
        """
            Create a function.

            **Parameters**

            .. important::
                #. **meta_props**:  meta data or name of the function. To see available meta names use:\n
                                    >>> client.repository.FunctionMetaNames.show()

                       **type**: dict\n
                #. **function**:  path to file with archived function content or function (as described above)\n
                         - As a 'function' may be used one of the following:\n
                         - filepath to gz file\n
                         - 'score' function reference, where the function is the function which will be deployed\n
                         - generator function, which takes no argument or arguments which all have primitive python default values and as result return 'score' function\n
                   **type**: str or function\n

            **Output**

            .. important::

                    **returns**: Metadata of the function created.\n
                    **return type**: dict\n

            **Example**

                 The most simple use is (using `score` function):\n

                 >>> meta_props = {
                 >>>    client.repository.FunctionMetaNames.NAME: "function",
                 >>>    client.repository.FunctionMetaNames.DESCRIPTION: "This is ai function",
                 >>>    client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID: "53dc4cf1-252f-424b-b52d-5cdd9814987f"}

                 >>> def score(payload):
                 >>>      values = [[row[0]*row[1]] for row in payload['values']]
                 >>>      return {'fields': ['multiplication'], 'values': values}
                 >>> stored_function_details = client.repository.store_function(score, meta_props)

                 Other, more interesting example is using generator function.
                 In this situation it is possible to pass some variables:

                    >>> wml_creds = {...}
                    >>> def gen_function(wml_credentials=wml_creds, x=2):
                    >>>        def f(payload):
                    >>>            values = [[row[0]*row[1]*x] for row in payload['values']]
                    >>>            return {'fields': ['multiplication'], 'values': values}
                    >>>        return f
                    >>> stored_function_details = client.repository.store_function(gen_function, meta_props)
            """

        return self._client._functions.store(function, meta_props)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_model_revision(self, model_uid):
        """
        Create a new version for a model.

        **Parameters**

        .. important::

            #. **model_uid**:  Model ID.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Model version details.\n
            **return type**: dict\n

        **Example**

            >>> stored_model_revision_details = client.repository.create_model_revision( model_uid="MODELID")
        """

        return self._client._models.create_revision(model_uid=model_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_pipeline_revision(self, pipeline_uid):
        """
        Create a new version for a model.

        **Parameters**

        .. important::

            #. **pipeline_uid**:  Unique ID of the Pipeline.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Pipeline version details.\n
            **return type**: dict\n

        **Example**

            >>> stored_pipeline_revision_details = client.repository.create_pipeline_revision( pipeline_uid)
        """

        return self._client.pipelines.create_revision(pipeline_uid=pipeline_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_function_revision(self, function_uid):
        """
        Create a new version for a function.

        **Parameters**

        .. important::

            #. **function_uid**:  Unique ID of the function.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Function version details.\n
            **return type**: dict\n

        **Example**

            >>> stored_function_revision_details = client.repository.create_function_revision( function_uid)
        """

        return self._client._functions.create_revision(function_uid=function_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_experiment_revision(self, experiment_uid):
        """
        Create a new version for a experiment.

        **Parameters**

        .. important::

            #. **experiment_uid**:  Unique ID of the experiment.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: experiment version details.\n
            **return type**: dict\n

        **Example**

            >>> stored_experiment_revision_details = client.repository.create_experiment_revision(experiment_uid)
        """

        return self._client.experiments.create_revision(experiment_id=experiment_uid)

    def update_model(self, model_uid, updated_meta_props=None, update_model=None):
        """
        Updates existing model metadata.

        **Parameters**

        .. important::

            #. **model_uid**: Unique id of model which definition should be updated\n
               **type**: str\n

            #. **updated_meta_props**: elements which should be changed, where keys are ConfigurationMetaNames\n
               **type**: dict\n

            #. **update_model**: archived model content file or path to directory containing archived model file which should be changed for specific model_uid.
               This parameters is valid only for CP4D 3.0.0.\n
               **type**: object or archived model content file\n

        **Output**

        .. important::

            **returns**: metadata of updated model\n
            **return type**: dict\n

        **Example 1**

         >>> metadata = {
         >>> client.repository.ModelMetaNames.NAME:"updated_model"
         >>> }
         >>> model_details = client.repository.update_model(model_uid, updated_meta_props=metadata)

        **Example 2**

         >>> metadata = {
         >>> client.repository.ModelMetaNames.NAME:"updated_model"
         >>> }
         >>> model_details = client.repository.update_model(model_uid, updated_meta_props=metadata, update_model="newmodel_content.tar.gz")
        """

        return self._client._models.update(model_uid, updated_meta_props, update_model)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update_experiment(self, experiment_uid, changes):
        """
                Updates existing experiment metadata.

                **Parameters**

                .. important::

                    #. **experiment_uid**: Unique of Id experiment which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::

                    **returns**: metadata of updated experiment\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.repository.ExperimentMetaNames.NAME:"updated_exp"
                 >>> }
                 >>> exp_details = client.repository.update_experiment(experiment_uid, changes=metadata)

        """
        if self._client.WSD:
            raise WMLClientError('Experiments APIs are not supported in IBM Watson Studio Desktop.')

        return self._client.experiments.update(experiment_uid, changes)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update_function(self, function_uid, changes, update_function=None):
        """
        Updates existing function metadata.

        **Parameters**

        .. important::

            #. **function_uid**: Unique Id of function which define what should be updated\n
               **type**: str\n

            #. **changes**: Elements which should be changed, where keys are ConfigurationMetaNames.\n
               **type**: dict\n

            #. **update_function**:  Path to file with archived function content or function which should be changed for specific function_uid.
               This parameters is valid only for CP4D 3.0.0.\n
               **type**: str or function\n

        **Output**

        .. important::

            **returns**: metadata of updated function\n
            **return type**: dict\n

        **Example**

            >>> metadata = {
            >>> client.repository.FunctionMetaNames.NAME:"updated_function"
            >>> }
            >>>
            >>> function_details = client.repository.update_function(function_uid, changes=metadata)
        """
        return self._client._functions.update(function_uid, changes, update_function)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update_pipeline(self, pipeline_uid, changes):
        """
                Updates existing pipeline metadata.

                **Parameters**

                .. important::

                    #. **pipeline_uid**:  Unique Id of pipeline which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::

                   **returns**: metadata of updated pipeline\n
                   **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.repository.PipelineMetanames.NAME:"updated_pipeline"
                 >>> }
                 >>> pipeline_details = client.repository.update_pipeline(pipeline_uid, changes=metadata)

        """

        return self._client.pipelines.update(pipeline_uid, changes)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update_space(self, space_uid, changes):
        """
                Updates existing space metadata.

                **Parameters**

                .. important::
                    #. **space_uid**:  Unique Id of space which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::
                    **returns**: metadata of updated space\n
                    **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.repository.SpacesMetaNames.NAME:"updated_space"
                 >>> }
                 >>> space_details = client.repository.update_space(space_uid, changes=metadata)
        """
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.update(space_uid, changes)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def load(self, artifact_uid):
        """
           Load model from repository to object in local environment.

           **Parameters**

           .. important::

                #. **artifact_uid**:  Unique Id of model\n
                   **type**: str\n

           **Output**

           .. important::

                **returns**: model object\n
                **return type**: object\n

           **Example**

            >>> model_obj = client.repository.load(model_uid)
        """

        return self._client._models.load(artifact_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def download(self, artifact_uid, filename='downloaded_artifact.tar.gz', rev_uid=None, format=None):
        """
               Downloads configuration file for artifact with specified uid.

               **Parameters**

               .. important::

                    #. **artifact_uid**:  Unique Id of model, function, runtime or library\n
                       **type**: str\n

                    #. **filename**:  Name of the file to which the artifact content has to be downloaded\n
                       **default value**: downloaded_artifact.tar.gz\n
                       **type**: str\n

               **Output**

               .. important::

                   **returns**: Path to the downloaded artifact content\n
                   **return type**: str\n

               .. note::

                    If filename is not specified, the default filename is "downloaded_artifact.tar.gz".\n

               **Example**

                >>> client.repository.download(model_uid, 'my_model.tar.gz')
        """
        self._validate_type(artifact_uid, 'artifact_uid', STR_TYPE, True)
        self._validate_type(filename, 'filename', STR_TYPE, True)

        res = self._check_artifact_type(artifact_uid)

        if res['model'] is True:
            return self._client._models.download(artifact_uid, filename, rev_uid,format)
        elif res['function'] is True:
            return self._client._functions.download(artifact_uid, filename, rev_uid)
        elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_35 and res['library'] is True:
            return self._client.runtimes.download_library(artifact_uid, filename)
        elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_35 and res['runtime'] is True:
            return self._client.runtimes.download_configuration(artifact_uid, filename)
        else:
            raise WMLClientError('Unexpected type of artifact to download or Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid) )

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, artifact_uid):
        """
           Delete model, experiment, pipeline, space, runtime, library or function from repository.

           **Parameters**

           .. important::
                #. **artifact_uid**:  Unique id of stored model, experiment, function, pipeline, space, library or runtime \n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

           **Example**

            >>> client.repository.delete(artifact_uid)
        """

        artifact_uid = str_type_conv(artifact_uid)
        Repository._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and self._if_deployment_exist_for_asset(artifact_uid):
            raise WMLClientError(
                u'Cannot delete artifact that has existing deployments. Please delete all associated deployments and try again')
        params = self._client._params()
        if Repository.cloud_platform_spaces or self._client.ICP_PLATFORM_SPACES:
            # ideally purge_on_delete=true query param has to be provided for deletion of cams assets
            # This doesn't seem to be done for CP4D 3.0.1 and before. We should do this for CP4D 3.5
            params.update({'purge_on_delete': 'true'})

        if not self._ICP:
            response = requests.delete(self._href_definitions.get_asset_href(artifact_uid),
                                           params=params,
                                           headers=self._client._get_headers())
        else:
            response = requests.delete(self._href_definitions.get_asset_href(artifact_uid),
                                       params=params,
                                       headers=self._client._get_headers(), verify=False)

        if response.status_code == 200 or response.status_code == 204:
            if response.status_code == 200:
                response = self._handle_response(200, u'delete assets', response)
                return response
            else:
                response = self._handle_response(204, u'delete assets', response)
                return response
        else:
            if Repository.cloud_platform_spaces or self._client.ICP_PLATFORM_SPACES:
                # Since we are using /v2/assets for deletion, don't need all the logic
                # in the following else block. The else block is applicable only for cloud beta
                # and has to be kept till then. For 3.5, move logic to same as cloud convergence
                # for deletion
                if response.status_code == 404:
                    raise WMLClientError(u'Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid))
                else:
                    raise WMLClientError("Deletion error for the given id : ", response.text)
            else:
                artifact_type = self._check_artifact_type(artifact_uid)
                self._logger.debug(u'Attempting deletion of artifact with type: \'{}\''.format(str(artifact_type)))
                if self._client.WSD:
                    if artifact_type[u'model'] is True:
                        return self._client._models.delete(artifact_uid)
                    elif artifact_type[u'pipeline'] is True:
                        return self._client.pipelines.delete(artifact_uid)
                    elif artifact_type[u'function'] is True:
                        return self._client._functions.delete(artifact_uid)
                    else:
                        raise WMLClientError(u'Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid))
                else:
                    if artifact_type[u'model'] is True:
                        return self._client._models.delete(artifact_uid)
                    elif artifact_type[u'experiment'] is True:
                        return self._client.experiments.delete(artifact_uid)
                    elif artifact_type[u'pipeline'] is True:
                        return self._client.pipelines.delete(artifact_uid)
                    elif artifact_type[u'function'] is True:
                        return self._client._functions.delete(artifact_uid)
                    elif artifact_type[u'space'] is True:
                        return self._client.spaces.delete(artifact_uid)
                    elif artifact_type[u'runtime'] is True:
                        return self._client.runtimes.delete(artifact_uid)
                    elif artifact_type[u'library'] is True:
                        return self._client.runtimes.delete_library(artifact_uid)
                    else:
                        raise WMLClientError(u'Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid))

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, artifact_uid=None):
        """
           Get metadata of stored artifacts. If artifact_uid is not specified returns all models, experiments, functions, pipelines, spaces, libraries and runtimes metadata.

           **Parameters**

           .. important::
                #. **artifact_uid**: Unique Id of stored model, experiment, function, pipeline, space, library or runtime (optional)\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: stored artifact(s) metadata\n
                **return type**: dict\n
                dict (if artifact_uid is not None) or {"resources": [dict]} (if artifact_uid is None)\n

           .. note::
                If artifact_uid is not specified, all models, experiments, functions, pipelines, spaces, libraries and runtimes metadata is fetched\n

           **Example**

            >>> details = client.repository.get_details(artifact_uid)
            >>> details = client.repository.get_details()
        """

        artifact_uid = str_type_conv(artifact_uid)
        Repository._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, False)

        if artifact_uid is None and self._client.WSD is None:
                model_details = self._client._models.get_details()
                experiment_details = self.get_experiment_details()
                pipeline_details = self.get_pipeline_details()
                function_details = self._client._functions.get_details()

                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    space_details = self._client.spaces.get_details()
                    library_details = self._client.runtimes.get_library_details()
                    runtime_details = self._client.runtimes.get_details()
                    details = {
                        u'models': model_details,
                        u'experiments': experiment_details,
                        u'pipeline': pipeline_details,
                        u'runtimes': runtime_details,
                        u'libraries': library_details,
                        u'spaces': space_details,
                        u'functions': function_details
                    }
                else:
                    details = {
                        u'models': model_details,
                        u'experiments': experiment_details,
                        u'pipeline': pipeline_details,
                        u'functions': function_details
                    }
        else:
            if self._client.WSD and artifact_uid is None:
                raise WMLClientError(
                        u' artifiact_uid is mandatory for get_details() in IBM Watson Studio Desktop.')
            uid_type = self._check_artifact_type(artifact_uid)
            if self._client.WSD:
                if uid_type[u'model'] is True:
                    details = self._client._models.get_details(artifact_uid)
                elif uid_type[u'pipeline'] is True:
                    details = self.get_pipeline_details(artifact_uid)
                elif uid_type[u'function'] is True:
                    details = self._client._functions.get_details(artifact_uid)
                else:
                    raise WMLClientError(
                        u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))
            else:
                if uid_type[u'model'] is True:
                    details = self._client._models.get_details(artifact_uid)
                elif uid_type[u'experiment'] is True:
                    details = self.get_experiment_details(artifact_uid)
                elif uid_type[u'pipeline'] is True:
                    details = self.get_pipeline_details(artifact_uid)
                elif uid_type[u'function'] is True:
                    details = self._client._functions.get_details(artifact_uid)
                elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and uid_type[u'runtime'] is True:
                    details = self._client.runtimes.get_details(artifact_uid)
                elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and uid_type[u'library'] is True:
                    details = self._client.runtimes.get_library_details(artifact_uid)
                elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and uid_type[u'space'] is True:
                    details = self._client.spaces.get_details(artifact_uid)
                else:
                    raise WMLClientError(u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))

        return details

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_model_details(self, model_uid=None, limit=None):
        """
           Get metadata of stored model. If model_uid is not specified returns all models metadata.

           **Parameters**

           .. important::
                #. **model_uid**: Unique Id of Model (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of model(s)\n
                **return type**: dict (if model_uid is not None) or {"resources": [dict]} (if model_uid is None)\n

           .. note::
                If model_uid is not specified, all models metadata is fetched\n

           **Example**

            >>> model_details = client.repository.get_model_details(model_uid)
            >>> models_details = client.repository.get_model_details()
        """

        return self._client._models.get_details(model_uid, limit)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_model_revision_details(self, model_uid, rev_uid):
        """
           Get metadata of model revision. 

           **Parameters**

           .. important::
                #. **experiment_uid**:  Unique Id of model\n
                   **type**: str\n
                #. **limit**:  Unique id of model revision\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: model revision metadata\n
                **return type**: dict\n

           **Example**

                 >>> model_rev_details = client.respository.get_model_revision_details(model_uid, rev_uid)

        """

        if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError('Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data for Data 3.0 and above.')
        return self._client._models.get_revision_details(model_uid, rev_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_experiment_details(self, experiment_uid=None, limit=None):
        """
           Get metadata of experiment. If no experiment_uid is specified all experiments metadata is returned.

           **Parameters**

           .. important::
                #. **experiment_uid**: Unique Id of experiment (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: experiment(s) metadata\n
                **return type**: dict\n
                dict (if experiment_uid is not None) or {"resources": [dict]} (if experiment_uid is None)\n

           .. note::
                If experiment_uid is not specified, all experiments metadata is fetched\n

           **Example**

                 >>> experiment_details = client.respository.get_experiment_details(experiment_uid)

         """

        if self._client.WSD:
            raise WMLClientError('Experiment APIs are not supported in IBM Watson Studio Desktop.')

        experiment_uid = str_type_conv(experiment_uid)
        Repository._validate_type(experiment_uid, u'experiment_uid', STR_TYPE, False)
        Repository._validate_type(limit, u'limit', int, False)

        url = self._href_definitions.get_experiments_href()

        return self._client.experiments.get_details(experiment_uid, limit)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_experiment_revision_details(self, experiment_uid, rev_id):
        """
           Get metadata of experiment revision. 

           **Parameters**

           .. important::
                #. **experiment_uid**: Unique Id of experiment\n
                   **type**: str\n
                #. **rev_id**:  Unique id of experiment revision\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: experiment revision metadata\n
                **return type**: dict\n

           **Example**

                 >>> experiment_rev_details = client.respository.get_experiment__revision_details(experiment_uid, rev_uid)

        """

        if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data for Data 3.0 and above.')

        return self._client.experiments.get_revision_details(experiment_uid, rev_id)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_function_details(self, function_uid=None, limit=None):
        """

           Get metadata of function. If no function_uid is specified all functions metadata is returned.

           **Parameters**

           .. important::
                #. **function_uid**:  Unique Id of function (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: function(s) metadata\n
                **return type**: dict (if function_uid is not None) or {"resources": [dict]} (if function_uid is None)\n

           .. note::
                If function_uid is not specified, all functions metadata is fetched\n

           **Example**

                >>> function_details = client.respository.get_function_details(function_uid)
                >>> function_details = client.respository.get_function_details()
         """
        Repository._validate_type(function_uid, u'function_uid', STR_TYPE, False)
        Repository._validate_type(limit, u'limit', int, False)
        return self._client._functions.get_details(function_uid, limit)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_function_revision_details(self, function_uid, rev_id):
        """

           Get metadata of function revision.

           **Parameters**

           .. important::
                #. **function_uid**:  Unique Id of function\n
                   **type**: str\n
                #. **rev_id**:  Unique Id of function revision\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: function revision metadata\n
                **return type**: dict\n

           **Example**

                >>> function_rev_details = client.respository.get_function_revision_details(function_uid, rev_id)

        """

        if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError('Not supported in this release')
        return self._client._functions.get_revision_details(function_uid, rev_id)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_pipeline_details(self, pipeline_uid=None, limit=None):
        """
           Get metadata of stored pipelines. If pipeline_uid is not specified returns all pipelines metadata.

           **Parameters**

           .. important::
                #. **pipeline_uid**: Unique id of Pipeline(optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of pipeline(s)\n
                **return type**: dict (if pipeline_uid is not None) or {"resources": [dict]} (if pipeline_uid is None)\n

           .. note::
                If pipeline_uid is not specified, all pipelines metadata is fetched\n

           **Example**

                >>> pipeline_details = client.repository.get_pipeline_details(pipeline_uid)
                >>> pipeline_details = client.repository.get_pipeline_details()
        """

        Repository._validate_type(pipeline_uid, u'pipeline_uid', STR_TYPE, False)
        Repository._validate_type(limit, u'limit', int, False)
        return self._client.pipelines.get_details(pipeline_uid, limit)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_pipeline_revision_details(self, pipeline_uid, rev_id):
        """
           Get metadata of stored pipeline revision. 

           **Parameters**

           .. important::
                #. **pipeline_uid**: Unique id of Pipeline\n
                   **type**: str\n
                #. **rev_id**:  Unique id Pipeline revision\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: metadata of revision pipeline(s)\n
                **return type**: dict\n

           **Example**

                >>> pipeline_rev_details = client.repository.get_pipeline_revision_details(pipeline_uid, rev_id)
        """

        if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data for Data 3.0 and above.')
        return self._client.pipelines.get_revision_details(pipeline_uid, rev_id)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_space_details(self, space_uid=None, limit=None):
        """
           Get metadata of stored space. If space_uid is not specified returns all model spaces metadata.

           **Parameters**

           .. important::
                #. **space_uid**: Unique id of Space (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of space(s)\n
                **return type**: dict (if space_uid is not None) or {"resources": [dict]} (if space_uid is None)\n

           .. note::
                If space_uid is not specified, all spaces metadata is fetched\n

           **Example**

            >>> space_details = client.repository.get_space_details(space_uid)
            >>> space_details = client.repository.get_space_details()
        """

        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        Repository._validate_type(space_uid, u'space_uid', STR_TYPE, False)
        Repository._validate_type(limit, u'limit', int, False)
        return self._client.spaces.get_details(space_uid, limit)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_members_details(self, space_uid, member_id=None, limit=None):
        """
           Get metadata of members associated with a space. If member_uid is not specified, it returns all the members metadata.

           **Parameters**

           .. important::
                #. **space_uid**: Unique id of member (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of member(s) of a space\n
                **return type**: dict (if member_id is not None) or {"resources": [dict]} (if member_id is None)\n

           .. note::
                If member id is not specified, all members metadata is fetched\n

           **Example**

            >>> member_details = client.repository.get_member_details(space_uid,member_id)
        """

        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.get_members_details(space_uid,member_id)


    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_model_href(model_details):
        """
            Get href of stored model.

           **Parameters**

           .. important::
                #. **model_details**:  Metadata of the stored model\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of stored model\n
                **return type**: str\n

           **Example**

            >>> model_details = client.repository.get_model_detailsf(model_uid)
            >>> model_uid = client.repository.get_model_href(model_details)
        """

        return Models.get_href(model_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_model_uid(model_details):
        """
            Get Unique Id of stored model.

           **Parameters**

           .. important::
                #. **model_details**:  Metadata of the stored model\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: Unique Id of stored model\n
                **return type**: str\n

           **Example**

            >>> model_details = client.repository.get_model_details(model_uid)
            >>> model_uid = client.repository.get_model_uid(model_details)
        """

        return Models.get_id(model_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_model_id(model_details):
        """
            Get Unique Id of stored model.

           **Parameters**

           .. important::
                #. **model_details**:  Metadata of the stored model\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: Unique Id of stored model\n
                **return type**: str\n

           **Example**

            >>> model_details = client.repository.get_model_details(model_uid)
            >>> model_uid = client.repository.get_model_id(model_details)
        """

        return Models.get_id(model_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_experiment_uid(experiment_details):
        """
           Get Unique Id of stored experiment.

           **Parameters**

           .. important::
                #. **experiment_details**:  Metadata of the stored experiment\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: Unique Id of stored experiment\n
                **return type**: str\n

           **Example**

            >>> experiment_details = client.repository.get_experiment_detailsf(experiment_uid)
            >>> experiment_uid = client.repository.get_experiment_uid(experiment_details)

        """

        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return Experiments.get_uid(experiment_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_experiment_id(experiment_details):
        """
           Get Unique Id of stored experiment.

           **Parameters**

           .. important::
                #. **experiment_details**:  Metadata of the stored experiment\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: Unique Id of stored experiment\n
                **return type**: str\n

           **Example**

            >>> experiment_details = client.repository.get_experiment_details(experiment_uid)
            >>> experiment_uid = client.repository.get_experiment_id(experiment_details)

        """

        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return Experiments.get_id(experiment_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_experiment_href(experiment_details):
        """
           Get href of stored experiment.

           **Parameters**

           .. important::
                #. **experiment_details**:  Metadata of the stored experiment\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of stored experiment\n
                **return type**: str\n

           **Example**

             >>> experiment_details = client.repository.get_experiment_detailsf(experiment_uid)
             >>> experiment_href = client.repository.get_experiment_href(experiment_details)

        """
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return Experiments.get_href(experiment_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_function_id(function_details):
        """
            Get Id of stored function.

            **Parameters**

            .. important::
                #. **function_details**:  Metadata of the stored function\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Id of stored function\n
                **return type**: str\n

            **Example**

             >>> function_details = client.repository.get_function_details(function_uid)
             >>> function_id = client.repository.get_function_id(function_details)
        """
        return Functions.get_id(function_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_function_uid(function_details):
        """
            Get Unique Id of stored function. Deprecated!! Use get_function_id(function_details) instead

            **Parameters**

            .. important::
                #. **function_details**:  Metadata of the stored function\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Unique Id of stored function\n
                **return type**: str\n

            **Example**

             >>> function_details = client.repository.get_function_detailsf(function_uid)
             >>> function_uid = client.repository.get_function_uid(function_details)
        """
        return Functions.get_uid(function_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_pipeline_uid(pipeline_details):
        """
            Get pipeline_uid from pipeline details.

            **Parameters**

            .. important::
                #. **pipeline_details**:  Metadata of the stored pipeline\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Unique Id of pipeline\n
                **return type**: str

            **Example**

             >>> pipeline_details = client.repository.get_pipeline_details(pipeline_uid)
             >>> pipeline_uid = client.repository.get_pipeline_uid(pipeline_details)

        """

        return Pipelines.get_uid(pipeline_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_function_href(function_details):
        """

            Get href of stored function.

            **Parameters**

            .. important::
                #. **function_details**:  Metadata of the stored function\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: href of stored function\n
                **return type**: str\n

            **Example**

             >>> function_details = client.repository.get_function_detailsf(function_uid)
             >>> function_url = client.repository.get_function_href(function_details)
        """
        return Functions.get_href(function_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_pipeline_href(pipeline_details):
        """
            Get pipeline_hef from pipeline details.

            **Parameters**

            .. important::
                #. **pipeline_details**:  Metadata of the stored pipeline\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: pipeline href\n
                **return type**: str

            **Example**

             >>> pipeline_details = client.repository.get_pipeline_details(pipeline_uid)
             >>> pipeline_href = client.repository.get_pipeline_href(pipeline_details)
        """

        return Pipelines.get_href(pipeline_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_pipeline_id(pipeline_details):
        """
            Get pipeline_uid from pipeline details.

            **Parameters**

            .. important::
                #. **pipeline_details**:  Metadata of the stored pipeline\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Unique Id of pipeline\n
                **return type**: str

            **Example**

             >>> pipeline_details = client.repository.get_pipeline_details(pipeline_uid)
             >>> pipeline_uid = client.repository.get_pipeline_id(pipeline_details)

        """

        return Pipelines.get_id(pipeline_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_space_uid(space_details):
        """
            Get space_uid from space details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored space\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: Unique Id of space\n
                **return type**: str

            **Example**

             >>> space_details = client.repository.get_space_details(space_uid)
             >>> space_uid = client.repository.get_space_uid(space_details)
        """
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_uid(space_details)

    @staticmethod
    def get_member_uid(member_details):
        """
            Get member_uid from member details.

            **Parameters**

            .. important::
                #. **member_details**:  Metadata of the created member\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: unique id of member\n
                **return type**: str

            **Example**

             >>> member_details = client.repository.get_member_details(member_id)
             >>> member_id = client.repository.get_member_uid(member_details)
        """
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_member_uid(member_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_space_href(space_details):
        """
            Get space_href from space details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored space\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: space href\n
                **return type**: str

            **Example**

             >>> space_details = client.repository.get_space_details(space_uid)
             >>> space_href = client.repository.get_space_href(space_details)
        """
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_href(space_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_member_href(member_details):
        """
            Get member_href from member details.

            **Parameters**

            .. important::
                #. **space_details**:  Metadata of the stored member\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: member href\n
                **return type**: str

            **Example**

             >>> member_details = client.repository.get_member_details(member_id)
             >>> member_href = client.repository.get_member_href(member_details)
        """
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_member_href(member_details)

    def list(self):
        """
           List stored models, pipelines, runtimes, libraries, functions, spaces and experiments. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all models, pipelines, runtimes, libraries, functions, spaces and experiments in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list()
        """

        from tabulate import tabulate

        headers = self._client._get_headers()
        params = self._client._params()
        params.update({u'limit': 1000})
        #params = {u'limit': 1000} # TODO - should be unlimited, if results not sorted

        pool = Pool(processes=4)
        isIcp = self._ICP
        if self._client.WSD:
            raise WMLClientError(
                u'list() - Listing all artifact is not supported for IBM Watson Studio Desktop. '
                u'Use list method of specific artifact.')


        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            endpoints = {
                u'model': self._href_definitions.get_published_models_href(),
                u'experiment': self._href_definitions.get_experiments_href(),
                u'pipeline': self._href_definitions.get_pipelines_href(),
                u'function': self._href_definitions.get_functions_href()
            }
        else:
            endpoints = {
                u'model': self._href_definitions.get_published_models_href(),
                u'experiment': self._href_definitions.get_experiments_href(),
                u'pipeline': self._href_definitions.get_pipelines_href(),
                u'function': self._href_definitions.get_functions_href(),
                u'runtime': self._href_definitions.get_runtimes_href(),
                u'library': self._href_definitions.get_custom_libraries_href()
            }

        artifact_get = {}
        for artifact in endpoints:
            if (artifact=="library" or artifact=="runtime" or artifact=="space"):
                params = None
            else:
                params = self._client._params()
            artifact_get[artifact] = pool.apply_async(get_url,
                                                (endpoints[artifact], self._client._get_headers(), params, isIcp))

        # artifact_get = {artifact: pool.apply_async(get_url, (endpoints[artifact], headers, self._client._params(), isIcp)) for
        #           artifact in endpoints if (artifact != "library" or artifact != "runtime" or artifact != "space")}
        # artifact_no_space = {artifact: pool.apply_async(get_url, (endpoints[artifact], headers, None, isIcp)) for artifact
        #                    in endpoints if (artifact == "library" or artifact == "runtime")}
        # artifact_get.update(artifact_no_space)

        resources = {artifact: [] for artifact in endpoints}

        for artifact in endpoints:
            try:
                    response = artifact_get[artifact].get()
                    response_text = self._handle_response(200, u'getting all {}s'.format(artifact), response)
                    resources[artifact] = response_text[u'resources']
            except Exception as e:
                    self._logger.error(e)

        pool.close()

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            model_values = [(m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at'],
                             m[u'entity'][u'type'], u'model') for m in resources[u'model']]
            experiment_values = [
                (m[u'metadata'][u'id'], m[u'metadata'][u'name'], m['metadata']['created_at'], u'-', u'experiment') for m
                in resources[u'experiment']]
            pipeline_values = [
                (m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at'], u'-', u'pipeline') for m
                in self._client.pipelines.get_details()[u'resources']]
            function_values = [(m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at'], u'-',
                                m[u'entity'][u'type'] + u' function') for m in resources[u'function']]
            values = list(set(model_values + experiment_values + pipeline_values + function_values))
        else:
            model_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], m[u'entity'][u'type'], u'model') for m in resources[u'model']]
            experiment_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'],m['metadata']['created_at'], u'-', u'experiment') for m in resources[u'experiment']]
            pipeline_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', u'pipeline')for m in self._client.pipelines.get_details()[u'resources']]
            function_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', m[u'entity'][u'type'] + u' function') for m in resources[u'function']]
            runtime_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', m[u'entity'][u'platform'][u'name'] + u' runtime') for m in resources[u'runtime']]
            library_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', m[u'entity'][u'platform'][u'name'] + u' library') for m in resources[u'library']]
            values = list(set(model_values + experiment_values + pipeline_values + function_values + runtime_values + library_values))
        values = sorted(sorted(values, key=lambda x: x[2], reverse=True), key=lambda x: x[4])

        table = tabulate([[u'GUID', u'NAME', u'CREATED', u'FRAMEWORK', u'TYPE']] + values[:_DEFAULT_LIST_LENGTH])
        print(table)
        if len(values) > _DEFAULT_LIST_LENGTH:
             print('Note: Only first {} records were displayed. To display more use more specific list functions.'.format(_DEFAULT_LIST_LENGTH))


    def list_models(self, limit=None):
        """
           List stored models. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all models in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_models()
        """

        self._client._models.list(limit=limit)

    def list_experiments(self, limit=None):
        """
           List stored experiments. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all experiments in a table format.\n
                **return type**: None\n


           **Example**

            >>> client.repository.list_experiments()
        """
        if self._client.WSD:
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        self._client.experiments.list(limit=limit)

    def list_spaces(self, limit=None):
        """
           List stored spaces. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all spaces in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_spaces()
        """

        if self._client.WSD:
            raise WMLClientError('list_spaces - Listing spaces is not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        self._client.spaces.list(limit=limit)

    def list_functions(self, limit=None):
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

             >>> client.respository.list_functions()
        """
        self._client._functions.list(limit=limit)

    def list_pipelines(self, limit=None):
        """
           List stored pipelines. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all pipelines in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_pipelines()
        """

        self._client.pipelines.list(limit=limit)
    def list_members(self, space_uid ,limit=None):
            """
               List stored members of a space. If limit is set to None there will be only first 50 records shown.

               **Parameters**

               .. important::
                    #. **limit**:  limit number of fetched records\n
                       **type**: int\n

               **Output**

               .. important::
                    This method only prints the list of all members associated with a space in a table format.\n
                    **return type**: None\n

               **Example**

                >>> client.spaces.list_members()
            """
            if self._client.WSD:
                raise WMLClientError('list_members - Listing members is not supported for Watson Studio Desktop.')

            if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
                raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

            self._client.spaces.list_members(space_uid=space_uid,limit=limit)

    def _check_artifact_type(self, artifact_uid):
        artifact_uid = str_type_conv(artifact_uid)
        Repository._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, True)

        def _artifact_exists(response):
            return (response is not None) and (u'status_code' in dir(response)) and (response.status_code == 200)

        pool = Pool(processes=4)
        #headers =

        isIcp=self._ICP

        if self._client.WSD:
            endpoint = self._href_definitions.get_model_definition_assets_href() + "/" + artifact_uid
            response = requests.get(
                endpoint,
                params=self._client._params(),
                verify=False
            )

           # requestsget_url, (endpoint, self._client._get_headers(), self._client._params(), True))
            response_get = _artifact_exists(response)

            artifact_type = artifact_uid.rsplit(".")[0]
            artifact_list = ['wml_model', 'wml_pipeline', 'wml_function']
            artifact_type_exists = {artifact.rsplit('_')[-1]: (response_get and artifact == artifact_type) for artifact in artifact_list}
            return artifact_type_exists

        else:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                endpoints = {
                    u'model': self._href_definitions.get_model_last_version_href(artifact_uid),
                    u'pipeline': self._href_definitions.get_pipeline_href(artifact_uid),
                    u'experiment': self._href_definitions.get_experiment_href(artifact_uid),
                    u'function': self._href_definitions.get_function_href(artifact_uid)
                }
            else:
                endpoints = {
                        u'model': self._href_definitions.get_model_last_version_href(artifact_uid),
                        u'pipeline': self._href_definitions.get_pipeline_href(artifact_uid),
                        u'experiment': self._href_definitions.get_experiment_href(artifact_uid),
                        u'function': self._href_definitions.get_function_href(artifact_uid),
                        u'runtime': self._href_definitions.get_runtime_href(artifact_uid),
                        u'library': self._href_definitions.get_custom_library_href(artifact_uid),
                        u'space': self._href_definitions.get_space_href(artifact_uid)
                    }
            future = {}
            for artifact in endpoints:
                if (artifact=="library" or artifact=="runtime" or artifact=="space"):
                    params = None
                else:
                    params = self._client._params()
                future[artifact] = pool.apply_async(get_url, (endpoints[artifact], self._client._get_headers(), params , isIcp))

            # future_no_space = {artifact: pool.apply_async(get_url, (endpoints[artifact], headers, None, isIcp)) for artifact in endpoints if (artifact=="library" or artifact=="runtime" or artifact=="space")}
            # future.update(future_no_space)
            response_get = {artifact: None for artifact in endpoints}

            for artifact in endpoints:
                    try:
                        response_get[artifact] = future[artifact].get(timeout=180)
                        self._logger.debug(u'Response({})[{}]: {}'.format(endpoints[artifact], response_get[artifact].status_code, response_get[artifact].text))

                    except Exception as e:
                        self._logger.debug(u'Error during checking artifact type: ' + str(e))

            pool.close()
            artifact_type = {artifact: _artifact_exists(response_get[artifact]) for artifact in response_get}

            return artifact_type


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_revision(self, artifact_uid):
        """
        Create revision for passed artifact_uid.

        **Parameters**

        .. important::

            #. **artifact_uid**: Unique id of stored model, experiment, function or pipelines.\n
               **type**: str\n

        **Output**

        .. important::

            **returns**: Artifact new revision metadata.\n
            **return type**: dict\n

        **Example**

            >>> details = client.repository.create_revision(artifact_uid)
        """
        artifact_uid = str_type_conv(artifact_uid)
        Repository._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, True)

        uid_type = self._check_artifact_type(artifact_uid)
        if uid_type[u'experiment'] is True:
            return self._client.experiments.create_revision(artifact_uid)
        if uid_type[u'pipeline'] is True:
            return self._client.pipelines.create_revision(artifact_uid)
        else:
           raise WMLClientError(u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))

        return details


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def _get_revision_details(self, artifact_uid):
        """
           Get metadata of stored artifacts revisions.

           :param artifact_uid:  unique id of stored model or experiment or function or pipelines (optional)
           :type artifact_uid: {str_type}

           :returns: stored artifacts metadata
           :rtype: dict

           A way you might use me is:

           >>> details = client.repository.get_revision_details(artifact_uid)

        """
        artifact_uid = str_type_conv(artifact_uid)
        Repository._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, True)

        uid_type = self._check_artifact_type(artifact_uid)

        if uid_type[u'experiment'] is True:
            details = self._client.experiments.get_revision_details(artifact_uid)
        if uid_type[u'pipeline'] is True:
            details = self._client.pipelines.get_revisions(artifact_uid)

        else:
            raise WMLClientError(u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))
        return details

    def list_models_revisions(self, model_uid, limit=None):
        """
           List stored model revisions. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **model_uid**:  Uniquie Id of the model \n
                   **type**: str\n

                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all revisions of given model ID in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_models_revisions(model_uid)
        """

        self._client._models.list_revisions(model_uid, limit=limit)

    def list_pipelines_revisions(self, pipeline_uid, limit=None):
        """
           List stored pipeline revisions. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **model_uid**:  Uniquie Id of the pipeline \n
                   **type**: str\n
           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all revisions of given pipeline ID in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_pipelines_revisions(pipeline_uid)
        """
        self._client.pipelines.list_revisions(pipeline_uid, limit=limit)

    def list_functions_revisions(self, function_uid, limit=None):
        """
           List stored function revisions. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **function_uid**:  Uniquie Id of the function \n
                   **type**: str\n
           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all revisions of given function ID in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_functions_revisions(function_uid)
        """
        self._client._functions.list_revisions(function_uid, limit=limit)

    def list_experiments_revisions(self, experiment_uid, limit=None):
        """
           List stored experiment revisions. If limit is set to None there will be only first 50 records shown.

           **Parameters**

           .. important::
                #. **experiment_uid**:  Uniquie Id of the experiment \n
                   **type**: str\n

           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n

           **Output**

           .. important::
                This method only prints the list of all revisions of given experiment ID in a table format.\n
                **return type**: None\n

           **Example**

            >>> client.repository.list_experiments_revisions(experiment_uid)
        """
        self._client.experiments.list_revisions(experiment_uid, limit=limit)


