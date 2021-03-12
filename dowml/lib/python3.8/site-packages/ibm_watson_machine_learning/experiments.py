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
import time
import copy
from ibm_watson_machine_learning.wml_client_error import MissingValue, WMLClientError, MissingMetaProp
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from multiprocessing import Pool
from ibm_watson_machine_learning.utils import print_text_header_h1, print_text_header_h2, EXPERIMENT_DETAILS_TYPE,  format_metrics, STR_TYPE, STR_TYPE_NAME, docstring_parameter, group_metrics, str_type_conv, meta_props_str_conv
from ibm_watson_machine_learning.hpo import HPOParameter, HPOMethodParam
from ibm_watson_machine_learning.metanames import ExperimentMetaNames
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS

_DEFAULT_LIST_LENGTH = 50

class Experiments(WMLResource):
    """
       Run new experiment.

    """

    ConfigurationMetaNames = ExperimentMetaNames()
    """MetaNames for experiments creation."""

    @staticmethod
    def _HPOParameter(name, values=None, max=None, min=None, step=None):
        return HPOParameter(name, values, max, min, step)

    @staticmethod
    def _HPOMethodParam(name=None, value=None):
        return HPOMethodParam(name, value)

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._experiments_uids_cache = {}

    def store(self, meta_props):
        """
                Create an experiment.

                **Parameters**

                .. important::

                       #. **meta_props**:  meta data of the experiment configuration. To see available meta names use:\n
                                           >>> client.experiments.ConfigurationMetaNames.get()

                          **type**: dict\n

                **Output**

                .. important::

                       **returns**: stored experiment metadata\n
                       **return type**: dict\n

                **Example**

                    >>> metadata = {
                    >>>  client.experiments.ConfigurationMetaNames.NAME: 'my_experiment',
                    >>>  client.experiments.ConfigurationMetaNames.EVALUATION_METRICS: ['accuracy'],
                    >>>  client.experiments.ConfigurationMetaNames.TRAINING_REFERENCES: [
                    >>>      {
                    >>>        'pipeline': {'href': pipeline_href_1}
                    >>>
                    >>>      },
                    >>>      {
                    >>>        'pipeline': {'href':pipeline_href_2}
                    >>>      },
                    >>>   ]
                    >>> }
                    >>> experiment_details = client.experiments.store(meta_props=metadata)
                    >>> experiment_href = client.experiments.get_href(experiment_details)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)
        ##Check if default space is set
        if self._client.CAMS and not self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                metaProps['space'] = {'href': "/v4/spaces/"+self._client.default_space_id}
            elif self._client.default_project_id is not None:
                metaProps['project'] = {'href': "/v2/projects/"+self._client.default_project_id}
            else:
                raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                metaProps['space_id'] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                metaProps['project_id'] = self._client.default_project_id
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, STR_TYPE, True)
        else:
            self.ConfigurationMetaNames._validate(meta_props)


        if not self._ICP:
            if self._client.CLOUD_PLATFORM_SPACES:
                response_experiment_post = requests.post(
                    self._href_definitions.get_experiments_href(),
                    params=self._client._params(skip_for_create=True),
                    json=metaProps,
                    headers=self._client._get_headers()
                )
            else:
                response_experiment_post = requests.post(
                    self._href_definitions.get_experiments_href(),
                    json=metaProps,
                    headers=self._client._get_headers()
                )
        else:
            if self._client.ICP_PLATFORM_SPACES:
                response_experiment_post = requests.post(
                    self._href_definitions.get_experiments_href(),
                    json=metaProps,
                    params=self._client._params(skip_for_create=True),
                    headers=self._client._get_headers(),
                    verify=False
                )
            else:
                response_experiment_post = requests.post(
                    self._href_definitions.get_experiments_href(),
                    json=metaProps,
                    headers=self._client._get_headers(),
                    verify=False
                )

        return self._handle_response(201, u'saving experiment', response_experiment_post)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def update(self, experiment_uid, changes):
        """
                Updates existing experiment metadata.

                **Parameters**

                .. important::
                    #. **experiment_uid**:  UID of experiment which definition should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**

                .. important::

                     **returns**: metadata of updated experiment\n
                     **return type**: dict\n

                **Example**

                 >>> metadata = {
                 >>> client.experiments.ConfigurationMetaNames.NAME:"updated_exp"
                 >>> }
                 >>> exp_details = client.experiments.update(experiment_uid, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        experiment_uid = str_type_conv(experiment_uid)
        self._validate_type(experiment_uid, u'experiment_uid', STR_TYPE, True)
        self._validate_type(changes, u'changes', dict, True)
        meta_props_str_conv(changes)

        details = self._client.repository.get_details(experiment_uid)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes, with_validation=True)

        url = self._href_definitions.get_experiment_href(experiment_uid)
        if not self._ICP:
            response = requests.patch(url, json=patch_payload, params = self._client._params(),headers=self._client._get_headers())
        else:
            response = requests.patch(url, json=patch_payload, params = self._client._params(),headers=self._client._get_headers(), verify=False)
        updated_details = self._handle_response(200, u'experiment patch', response)

        return updated_details


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, experiment_uid=None, limit=None):
        """
                Get metadata of experiment(s). If no experiment UID is specified all experiments metadata is returned.

                **Parameters**

                .. important::

                       #. **experiment_uid**:  UID of experiment (optional)\n
                          **type**: str\n
                       #. **limit**:  limit number of fetched records (optional)\n
                          **type**: int\n

                **Output**

                .. important::

                    **returns**: experiment(s) metadata\n
                    **return type**: dict\n
                    dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

                .. note::

                   If UID is not specified, all experiments metadata is fetched\n

                **Example**

                     >>> experiment_details = client.experiments.get_details(experiment_uid)
                     >>> experiment_details = client.experiments.get_details()

         """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        experiment_uid = str_type_conv(experiment_uid)
        url = self._href_definitions.get_experiments_href()

        return self._get_artifact_details(url, experiment_uid, limit, 'experiment')

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(experiment_details):
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

                     >>> experiment_details = client.experiments.get_detailsf(experiment_uid)
                     >>> experiment_uid = client.experiments.get_uid(experiment_details)

        """
        Experiments._validate_type(experiment_details, u'experiment_details', object, True)
        if 'id' not in experiment_details[u'metadata']:
            Experiments._validate_type_of_details(experiment_details, EXPERIMENT_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                           [u'metadata', u'id'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(experiment_details):
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

                     >>> experiment_details = client.experiments.get_detailsf(experiment_id)
                     >>> experiment_uid = client.experiments.get_id(experiment_details)

        """
        Experiments._validate_type(experiment_details, u'experiment_details', object, True)
        if 'id' not in experiment_details[u'metadata']:
            Experiments._validate_type_of_details(experiment_details, EXPERIMENT_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                           [u'metadata', u'id'])

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(experiment_details):
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

                 >>> experiment_details = client.experiments.get_detailsf(experiment_uid)
                 >>> experiment_href = client.experiments.get_href(experiment_details)

        """
        Experiments._validate_type(experiment_details, u'experiment_details', object, True)
        if 'href' in experiment_details['metadata']:
            Experiments._validate_type_of_details(experiment_details, EXPERIMENT_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                           [u'metadata', u'href'])
        else:
            experiment_id = WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                                     [u'metadata', u'id'])
            return "/ml/v4/experiments/" + experiment_id


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, experiment_uid):
        """
            Delete a stored experiment.

            **Parameters**

            .. important::
                #. **experiment_uid**:  Unique Id of the stored experiment\n
                   **type**: str\n

            **Output**

            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n

            **Example**

                >>> client.experiments.delete(experiment_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        experiment_uid = str_type_conv(experiment_uid)
        Experiments._validate_type(experiment_uid, u'experiment_uid', STR_TYPE, True)

        url = self._href_definitions.get_experiment_href(experiment_uid)
        if not self._ICP:
            response = requests.delete(url, params = self._client._params(),headers=self._client._get_headers())
        else:
            response = requests.delete(url, params = self._client._params(),headers=self._client._get_headers(), verify=False)

        return self._handle_response(204, u'experiment deletion', response, False)


    def list(self, limit=None):
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

                >>> client.experiments.list()

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        experiment_resources = self.get_details(limit=limit)[u'resources']
        header_list = [u'GUID', u'NAME', u'CREATED']
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            experiment_values = [(m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                                 experiment_resources]
            header_list = [u'ID', u'NAME', u'CREATED']
        else:
            experiment_values = [(m[u'metadata'][u'id'], m[u'entity'][u'name'], m[u'metadata'][u'created_at']) for m in
                                 experiment_resources]

        self._list(experiment_values, header_list, limit, _DEFAULT_LIST_LENGTH)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def create_revision(self, experiment_id):
        """
        Creates a new experiment revision.
        :param experiment_id:
        :return:  stored experiment new revision details
        Example:

           >>> experiment_revision_artifact = client.experiments.create_revision(experiment_id)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        experiment_id = str_type_conv(experiment_id)
        Experiments._validate_type(experiment_id, u'experiment_id', STR_TYPE, True)

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revision support is not there in this WML server. It is supported only from 3.1.0 onwards.')
        else:
            url =self._href_definitions.get_experiments_href()
            return self._create_revision_artifact(url, experiment_id, 'experiments')

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_revision_details(self, experiment_uid, rev_uid):
        """
            Get metadata of stored experiments revisions.

            :param experiment_uid: stored experiment UID (optional)
            :type experiment_uid: {str_type}

            :param rev_id: rev_id number of experiment
            :type rev_id: int

            :returns: stored experiment revision metadata
            :rtype: dict

            Example:

            >>> experiment_details = client.repository.get_revision_details(experiment_uid,rev_id)

         """
        self._client._check_if_either_is_set()
        model_uid = str_type_conv(experiment_uid)
        Experiments._validate_type(experiment_uid, u'experiment_uid', STR_TYPE, True)
        Experiments._validate_type(rev_uid, u'rev_uid', int, True)
        if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data for Data 3.0 and above.')
        else:
            url = self._href_definitions.get_experiment_href(experiment_uid)
            return self._get_with_or_without_limit(url, limit=None, op_name="experiments",
                                                   summary=None, pre_defined=None, revision=rev_uid)

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def list_revisions(self, experiment_uid, limit=None):
        """
           List all revision for the given experiment uid.

           :param experiment_uid: Unique id of stored experiment.
           :type experiment_uid: {str_type}

           :param limit: limit number of fetched records (optional)
           :type limit: int

           :returns: stored experiment revisions details
           :rtype: table

           >>> client.experiments.list_revisions(experiment_uid)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        experiment_uid = str_type_conv(experiment_uid)

        Experiments._validate_type(experiment_uid, u'experiment_uid', STR_TYPE, True)

        if self._client.ICP_30 is None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revision support is not there in this WML server. It is supported only from 3.1.0 onwards.')
        else:
            url = self._href_definitions.get_experiment_href(experiment_uid)
            experiment_resources = self._get_artifact_details(url, "revisions", limit, 'model revisions')[u'resources']
            experiment_values = [
                (m[u'metadata'][u'rev'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                experiment_resources]

            self._list(experiment_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)


    def clone(self, experiment_uid, space_id=None, action="copy", rev_id=None):
        """
            Creates a new experiment identical with the given experiment either in the same space or in a new space. All dependent assets will be cloned too.

            **Parameters**

            .. important::
                #. **model_id**:  Guid of the experiment to be cloned:\n

                   **type**: str\n

                #. **space_id**: Guid of the space to which the experiment needs to be cloned. (optional)

                   **type**: str\n

                #. **action**: Action specifying "copy" or "move". (optional)

                   **type**: str\n

                #. **rev_id**: Revision ID of the experiment. (optional)

                   **type**: str\n

            **Output**

            .. important::

                    **returns**: Metadata of the experiment cloned.\n
                    **return type**: dict\n

            **Example**
             >>> client.experiments.clone(experiment_uid=artifact_id,space_id=space_uid,action="copy")

            .. note::
                * If revision id is not specified, all revisions of the artifact are cloned\n

                * Default value of the parameter action is copy\n

                * Space id is mandatory for move action\n

            """
        artifact = str_type_conv(experiment_uid)
        Experiments._validate_type(artifact, 'experiment_uid', STR_TYPE, True)
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

        url = self._href_definitions.get_experiment_href(experiment_uid)
        if not self._ICP:
            response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers())
        else:
            response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers(), verify=False)

        details = self._handle_response(expected_status_code=200, operationName=u'cloning experiment',
                                        response=response_post)

        return details


