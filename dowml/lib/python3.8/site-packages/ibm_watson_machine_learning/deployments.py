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
from ibm_watson_machine_learning.utils import DEPLOYMENT_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, print_text_header_h1, print_text_header_h2, STR_TYPE, STR_TYPE_NAME, docstring_parameter, str_type_conv, \
    StatusLogger, meta_props_str_conv, convert_metadata_to_parameters
from ibm_watson_machine_learning.wml_client_error import WMLClientError, MissingValue, NoVirtualDeploymentSupportedForICP
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.metanames import DeploymentMetaNames, ScoringMetaNames, DecisionOptimizationMetaNames, DeploymentNewMetaNames
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
import numpy as np
#from ibm_watson_machine_learning.metanames import DeploymentMetaNames, ScoreMetaNames, EnvironmentMetaNames

lib_checker = LibraryChecker()

class Deployments(WMLResource):
    """
        Deploy and score published artifacts (models and functions).

    """
    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.CLOUD_PLATFORM_SPACES and not client.ICP_PLATFORM_SPACES:
            Deployments._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Deployments._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self.session = requests.Session()
        self._ICP = client.ICP
        if client.CLOUD_PLATFORM_SPACES or client.ICP_PLATFORM_SPACES:
            self.ConfigurationMetaNames = DeploymentNewMetaNames()
        else:
            self.ConfigurationMetaNames = DeploymentMetaNames()

        if client.CLOUD_PLATFORM_SPACES:
            Deployments.cloud_platform_spaces = True

        if client.ICP_PLATFORM_SPACES:
            Deployments.icp_platform_spaces = True

        self.ScoringMetaNames = ScoringMetaNames()
        self.DecisionOptimizationMetaNames = DecisionOptimizationMetaNames()

    def _deployment_status_errors_handling(self, deployment_details, operation_name, deployment_id):
        try:
            if 'failure' in deployment_details['entity']['status']:
                errors = deployment_details[u'entity'][u'status'][u'failure'][u'errors']
                for error in errors:
                    if type(error) == str:
                        try:
                            error_obj = json.loads(error)
                            print(error_obj[u'message'])
                        except:
                            print(error)
                    elif type(error) == dict:
                        print(error['message'])
                    else:
                        print(error)
                raise WMLClientError('Deployment ' + operation_name + ' failed for deployment id: ' + deployment_id +'. Errors: ' + str(errors))
            else:
                print(deployment_details['entity']['status'])
                raise WMLClientError('Deployment ' + operation_name + ' failed for deployment id: ' + deployment_id +'. Error: ' + str(deployment_details['entity']['status']['state']))
        except WMLClientError as e:
            raise e
        except Exception as e:
            self._logger.debug('Deployment ' + operation_name + ' failed: ' + str(e))
            print(deployment_details['entity']['status']['failure'])
            raise WMLClientError('Deployment ' + operation_name + ' failed for deployment id: ' + deployment_id + '.')

    @docstring_parameter({'str_type': STR_TYPE_NAME})  # TODO model_uid and artifact_uid should be changed to artifact_uid only
    def create(self, artifact_uid=None, meta_props=None, rev_id=None, **kwargs):
            """
                Create a deployment from an artifact. As artifact we understand model or function which may be deployed.
                
                **Parameters**
                
                .. important::

                    #. **artifact_uid**:  Published artifact UID (model or function uid)\n
                       **type**: str\n

                    #. **meta_props**: metaprops. To see the available list of metanames use:\n
                                       >>> client.deployments.ConfigurationMetaNames.get()

                       **type**: dict\n
                
                **Output**
                
                .. important::

                    **returns**: metadata of the created deployment\n
                    **return type**: dict\n

                **Example**

                 >>> meta_props = {
                 >>> wml_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
                 >>> wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
                 >>> wml_client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : { "id":  "e7ed1d6c-2e89-42d7-aed5-8sb972c1d2b"}
                 >>> }
                 >>> deployment_details = client.deployments.create(artifact_uid, meta_props)

             """
            #artifact_uid = str_type_conv(artifact_uid) if artifact_uid is not None else (str_type_conv(kwargs['model_uid'] if 'model_uid' in kwargs else None))
            ##To be removed once deployments adds support for projects
            WMLResource._chk_and_block_create_update_for_python36(self)
            self._client._check_if_space_is_set()

            Deployments._validate_type(artifact_uid, u'artifact_uid', STR_TYPE, True)

            if self._ICP:
                predictionUrl = self._wml_credentials[u'url']

            if meta_props is None:
                raise WMLClientError("Invalid input. meta_props can not be empty.")

            if self._client.CLOUD_PLATFORM_SPACES and 'r_shiny' in meta_props:
                raise WMLClientError('Shiny is not supported in this release')

            metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)

            if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'wml_instance_id' in meta_props:
                metaProps.update({'wml_instance_id': meta_props['wml_instance_id']})

            if not self._client.ICP_30 and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                ## Remove new meataprop from older one, just in case if user provides.
                if self.ConfigurationMetaNames.HARDWARE_SPEC in metaProps:
                    metaProps.pop(self.ConfigurationMetaNames.HARDWARE_SPEC)
                if self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS in metaProps:
                    metaProps.pop(self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS)
                if self.ConfigurationMetaNames.R_SHINY in metaProps:
                    metaProps.pop(self.ConfigurationMetaNames.R_SHINY)

                artifact_details = self._client.repository.get_details(artifact_uid)
                artifact_href = artifact_details['metadata']['href']
                artifact_revision_id = artifact_href.split('=')[1]

                if 'model' in artifact_href:
                    if self._client.CAMS:
                        details = "/v4/models/"+artifact_uid + "?space_id=" + self._client.default_space_id
                    else:
                        details = "/v4/models/" + artifact_uid  + "?rev=" + artifact_revision_id
                elif 'function' in artifact_href:
                    if self._client.CAMS:
                        details = "/v4/functions/" + artifact_uid + "?space_id=" + self._client.default_space_id
                    else:
                        details = "/v4/functions/" + artifact_uid  + "?rev=" + artifact_revision_id
                else:
                    raise WMLClientError('Unexpected artifact type: \'{}\'.'.format(artifact_uid))
                if 'model' in artifact_href:
                    if "tensorflow_1.11" in artifact_details["entity"]["type"] or "tensorflow_1.5" in \
                            artifact_details["entity"]["type"]:
                        print("Note: Model of framework tensorflow and versions 1.5/1.11 has been deprecated. These versions will not be supported after 26th Nov 2019.")

                metaProps['asset'] = {'href': details}
                if self._client.CAMS:
                    if self._client.default_space_id is not None:
                        metaProps['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
                    else:
                        raise WMLClientError(
                            "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")

            ##Check if default space is set
            else:
                metaProps['asset'] = metaProps.get('asset') if metaProps.get('asset') else {'id': artifact_uid}
                if rev_id is not None:
                    metaProps['asset'].update({'rev': rev_id})

                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    metaProps['space_id'] = self._client.default_space_id
                else:
                    if 'space' not in metaProps:
                        if self._client.default_space_id is not None:
                            metaProps['space_id'] = self._client.default_space_id
                        else:
                            raise WMLClientError(
                                "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")

            url = self._href_definitions.get_deployments_href()
            if not self._ICP:
                if self._client.CLOUD_PLATFORM_SPACES:
                    response = requests.post(
                        url,
                        json=metaProps,
                        params=self._client._params(),  # version is mandatory
                        headers=self._client._get_headers())
                else:
                    response = requests.post(
                        url,
                        json=metaProps,
                        headers=self._client._get_headers())
            else:
                if self._client.ICP_PLATFORM_SPACES:
                    response = requests.post(
                        url,
                        json=metaProps,
                        params=self._client._params(),  # version is mandatory
                        headers=self._client._get_headers(),
                        verify=False)
                else:
                    response = requests.post(
                        url,
                        json=metaProps,
                        headers=self._client._get_headers(),
                        verify=False)

            ## Post Deployment call executed
            if response.status_code == 202:
                deployment_details = response.json()
                #print(deployment_details)

                if self._ICP:
                    if 'online_url' in deployment_details["entity"]["status"]:
                     scoringUrl = deployment_details.get(u'entity').get(u'status').get('online_url').get('url').replace('https://ibm-nginx-svc:443', predictionUrl)
                     deployment_details[u'entity'][u'status']['online_url']['url'] = scoringUrl

                deployment_uid = self.get_uid(deployment_details)

                import time
                print_text_header_h1(u'Synchronous deployment creation for uid: \'{}\' started'.format(artifact_uid))

                status = deployment_details[u'entity'][u'status']['state']

                notifications = []

                with StatusLogger(status) as status_logger:
                    while True:
                        time.sleep(5)
                        deployment_details = self._client.deployments.get_details(deployment_uid)
                        #this is wrong , needs to update for ICP
                        if "system" in deployment_details:
                            notification = deployment_details['system']['warnings'][0]['message']
                            if notification not in notifications:
                                print("\nNote: " + notification)
                                notifications.append(notification)
                        if self._ICP and not self._client.ICP_PLATFORM_SPACES:
                            scoringUrl = deployment_details.get(u'entity').get(u'asset').get('href').replace('https://wml-os-envoy:16600', predictionUrl)
                            deployment_details[u'entity'][u'asset']['href'] = scoringUrl
                        status = deployment_details[u'entity'][u'status'][u'state']
                        status_logger.log_state(status)
                        if status != u'DEPLOY_IN_PROGRESS' and status != "initializing":
                            break
                if status == u'DEPLOY_SUCCESS' or status == u'ready':
                    print(u'')
                    print_text_header_h2(
                        u'Successfully finished deployment creation, deployment_uid=\'{}\''.format(deployment_uid))
                    return deployment_details
                else:
                    print_text_header_h2(u'Deployment creation failed')
                    self._deployment_status_errors_handling(deployment_details, 'creation', deployment_uid)
            else:
                error_msg = u'Deployment creation failed'
                reason = response.text
                print(reason)
                print_text_header_h2(error_msg)
                raise WMLClientError(error_msg + '. Error: ' + str(response.status_code) + '. ' + reason)

            #return self._handle_response(202, u'created deployment', response)
    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_uid(deployment_details):
        """
            Get deployment_uid from deployment details. Deprecated!! Use get_id(deployment_details) instead
            
            **Parameters**
            
            .. important::
                #. **deployment_details**:  Metadata of the deployment\n
                   **type**: dict\n
            
            **Output**
            
            .. important::
                **returns**: deployment UID that is used to manage the deployment\n
                **return type**: str
            
            **Example**
            
             >>> deployment_uid = client.deployments.get_uid(deployment)
        """
        Deployments._validate_type(deployment_details, u'deployment_details', dict, True)

        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)

        try:
            if 'id' in deployment_details[u'metadata']:
                uid = deployment_details.get(u'metadata').get(u'id')
            else:
                uid = deployment_details.get(u'metadata').get(u'guid')
        except Exception as e:
            raise WMLClientError(u'Getting deployment UID from deployment details failed.', e)

        if uid is None:
            raise MissingValue(u'deployment_details.metadata.guid')

        return uid

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_id(deployment_details):
        """
            Get deployment id from deployment details.

            **Parameters**

            .. important::
                #. **deployment_details**:  Metadata of the deployment\n
                   **type**: dict\n

            **Output**

            .. important::
                **returns**: deployment ID that is used to manage the deployment\n
                **return type**: str

            **Example**

             >>> deployment_id = client.deployments.get_id(deployment)
        """

        return Deployments.get_uid(deployment_details)

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_href(deployment_details):
        """
            Get deployment_href from deployment details.
            
            **Parameters**
            
            .. important::
                #. **deployment_details**:  Metadata of the deployment.\n
                   **type**: dict\n
            
            **Output**
            
            .. important::
                **returns**: Deployment href that is used to manage the deployment.\n
                **return type**: str
            
            **Example**
            
            >>> deployment_href = client.deployments.get_href(deployment)

        """
        Deployments._validate_type(deployment_details, u'deployment_details', dict, True)
        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)

        try:
            if 'href' in deployment_details[u'metadata']:
                url = deployment_details.get(u'metadata').get(u'href')
            else:
                url = "/ml/v4/deployments/{}".format(deployment_details[u'metadata'][u'id'])
        except Exception as e:
            raise WMLClientError(u'Getting deployment url from deployment details failed.', e)

        if url is None:
            raise MissingValue(u'deployment_details.metadata.href')

        return url

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_details(self, deployment_uid=None, limit=None):
        """
           Get information about your deployment(s). If deployment_uid is not passed, all deployment details are fetched.
           
           **Parameters**
           
           .. important::
                #. **deployment_uid**: Unique Id of Deployment (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n
           
           **Output**
           
           .. important::
                **returns**: metadata of deployment(s)\n
                **return type**: dict\n
                dict (if deployment_uid is not None) or {"resources": [dict]} (if deployment_uid is None)\n

           .. note::
                If deployment_uid is not specified, all deployments metadata is fetched\n

           
           **Example**
           
            >>> deployment_details = client.deployments.get_details(deployment_uid)
            >>> deployment_details = client.deployments.get_details(deployment_uid=deployment_uid)
            >>> deployments_details = client.deployments.get_details()

        """
        ##To be removed once deployments adds support for deployments
        self._client._check_if_space_is_set()

        deployment_uid = str_type_conv(deployment_uid)
        Deployments._validate_type(deployment_uid, u'deployment_uid', STR_TYPE, False)

        if deployment_uid is not None and not is_uid(deployment_uid):
            raise WMLClientError(u'\'deployment_uid\' is not an uid: \'{}\''.format(deployment_uid))

        url = self._href_definitions.get_deployments_href()
        ## TODO:  Adding a work around till v4/deployment accepts space_id query parameter for GET
        if self._ICP and not self._client.ICP_PLATFORM_SPACES:
            if deployment_uid is not None:
                url = url + '/' + deployment_uid
                if self._client.ICP_35:
                    response = requests.get(
                        url,
                        params = self._client._params(),
                        headers=self._client._get_headers(),
                        verify=False
                    )
                else:
                    response = requests.get(
                        url,
                        headers=self._client._get_headers(),
                        verify=False
                    )
                return self._handle_response(200, "deployment", response)
            else:
                deployment_details = self._get_artifact_details(url, deployment_uid, limit, 'deployments')
                if "system" in deployment_details:
                    print("Note: " + deployment_details['system']['warnings'][0]['message'])
                return deployment_details
        else:
            details = self._get_artifact_details(url, deployment_uid, limit, 'deployments')
            
            return details

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_scoring_href(deployment_details):
        """
            Get scoring url from deployment details.
            
            **Parameters**
            
            .. important::
                #. **deployment_details**:  Metadata of the deployment\n
                   **type**: dict\n
            
            **Output**
            
            .. important::
                **returns**: scoring endpoint url that is used for making scoring requests\n
                **return type**: str\n
            
            **Example**
            
             >>> scoring_href = client.deployments.get_scoring_href(deployment)

        """

        Deployments._validate_type(deployment_details, u'deployment', dict, True)

        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)
        scoring_url = None
        try:
            url = deployment_details['entity']['status'].get('online_url')
            if url is not None:
                scoring_url = deployment_details['entity']['status']['online_url']['url']
            else:
                raise MissingValue(u'Getting scoring url for deployment failed. This functionality is  available only for sync deployments')

        except Exception as e:
            raise WMLClientError(u'Getting scoring url for deployment failed. This functionality is  available only for sync deployments', e)

        if scoring_url is None:
            raise MissingValue(u'scoring_url missing in online_predictions')
        return scoring_url

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def delete(self, deployment_uid):
        """
            Delete deployment.
            
            **Parameters**
            
            .. important::
                #. **deployment uid**:  Unique Id of  Deployment\n
                   **type**: str\n
            
            **Output**
            
            .. important::
                **returns**: status ("SUCCESS" or "FAILED")\n
                **return type**: str\n
            
            **Example**
            
             >>> client.deployments.delete(deployment_uid)

        """
        ##To be removed once deployments adds support for deployments
        self._client._check_if_space_is_set()

        deployment_uid = str_type_conv(deployment_uid)
        Deployments._validate_type(deployment_uid, u'deployment_uid', STR_TYPE, True)

        if deployment_uid is not None and not is_uid(deployment_uid):
            raise WMLClientError(u'\'deployment_uid\' is not an uid: \'{}\''.format(deployment_uid))

        deployment_url = self._href_definitions.get_deployment_href(deployment_uid)

        if not self._ICP:
            response_delete = requests.delete(
                deployment_url,
                params=self._client._params(),
                headers=self._client._get_headers())
        else:
            if self._client.ICP_PLATFORM_SPACES:
                response_delete = requests.delete(
                    deployment_url,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                    verify=False)
            else:
                response_delete = requests.delete(
                    deployment_url,
                    #TODO:  The below line needs to uncommeted after /v4/deployment accespts query param
                   # params=self._client._params(),
                    headers=self._client._get_headers(),
                    verify=False)

        return self._handle_response(204, u'deployment deletion', response_delete, False)


    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def score(self,deployment_id, meta_props):
        """
                    Make scoring requests against deployed artifact.

                    **Parameters**

                    .. important::
                        #. **deployment_id**:  Unique Id of the deployment to be scored\n
                           **type**: str\n
                        #. **meta_props**:  Meta props for scoring\n
                                            >>> Use client.deployments.ScoringMetaNames.show() to view the list of ScoringMetaNames.
                           **type**: dict\n
                        #. **transaction_id**:  transaction id to be passed with records during payload logging (optional)\n
                           **type**: str\n

                    **Output**

                    .. important::
                        **returns**: scoring result containing prediction and probability\n
                        **return type**: dict\n

                    .. note::

                            * client.deployments.ScoringMetaNames.INPUT_DATA is the only metaname valid for sync scoring.\n
                            * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

                    **Example**

                     >>> scoring_payload = {wml_client.deployments.ScoringMetaNames.INPUT_DATA:
                     >>>        [{'fields':
                     >>>            ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                     >>>            'values': [
                     >>>                ['M',23,'Single','Student'],
                     >>>                ['M',55,'Single','Executive']
                     >>>                ]
                     >>>         }
                     >>>       ]}
                     >>> predictions = client.deployments.score(deployment_id, scoring_payload)
                     >>> predictions = client.deployments.score(deployment_id, scoring_payload,async=True)

        """
        ##To be removed once deployments adds support for deployments
        self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, u'deployment_id', STR_TYPE, True)
        Deployments._validate_type(meta_props, u'meta_props', dict, True)

        if meta_props.get(self.ScoringMetaNames.INPUT_DATA) is None:
            raise WMLClientError("Scoring data input 'ScoringMetaNames.INPUT_DATA' is mandatory for synchronous "
                                 "scoring")

        scoring_data = meta_props[self.ScoringMetaNames.INPUT_DATA]

        if scoring_data is not None:
            score_payload = []
            for each_score_request in scoring_data:
                lib_checker.check_lib(lib_name='pandas')
                import pandas as pd

                scoring_values = each_score_request["values"]
                # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
                if (isinstance(scoring_values, pd.DataFrame)):
                    fields_names = scoring_values.columns.values.tolist()
                    values = scoring_values.values.tolist()

                    each_score_request["values"] = values
                    if fields_names is not None:
                        each_score_request["fields"] = fields_names


                ## If payload is a numpy dataframe

                elif (isinstance(scoring_values, np.ndarray)):

                    values = scoring_values.tolist()
                    each_score_request["values"] = values


                # else:
                #     payload_score["values"] = each_score_request["values"]
                #     if "fields" in each_score_request:
                #         payload_score["fields"] = each_score_request["fields"]
                score_payload.append(each_score_request)

            ##See if it is scoring or DecisionOptimizationJob

        payload = {}

        payload["input_data"] = score_payload

        headers = self._client._get_headers()

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            scoring_url = self._wml_credentials["url"] + "/ml/v4/deployments/" + deployment_id + "/predictions"
        else:
            scoring_url = self._wml_credentials["url"] + "/v4/deployments/" + deployment_id + "/predictions"

        if not self._ICP:
            if self._client.CLOUD_PLATFORM_SPACES:
                params = self._client._params()
                del params[u'space_id']
                response_scoring = self.session.post(
                    scoring_url,
                    json=payload,
                    params=params,  # version parameter is mandatory
                    headers=headers)
            else:
                response_scoring = self.session.post(
                    scoring_url,
                    json=payload,
                    headers=headers)
        else:
            if self._client.ICP_PLATFORM_SPACES:
                params = self._client._params()
                del params[u'space_id']
                response_scoring = self.session.post(
                    scoring_url,
                    json=payload,
                    params=params,  # version parameter is mandatory
                    headers=headers,
                    verify=False)
            else:
                response_scoring = self.session.post(
                    scoring_url,
                    json=payload,
                    headers=headers,
                    verify=False)

        return self._handle_response(200, u'scoring', response_scoring)

        #########################################

    @staticmethod
    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def get_download_url(deployment_details):
        """
        Get deployment_download_url from deployment details.

        **Parameters**

        .. important::

            #. **deployment_details**: Created deployment details.\n
               **type**: dict\n

        **Output**

        .. important::
            **returns**: Deployment download URL that is used to get file deployment (for example: Core ML).\n
            **return type**: str\n

        **Example**

            >>> deployment_url = client.deployments.get_download_url(deployment)
        """
        Deployments._validate_type(deployment_details, u'deployment_details', dict, True)
        try:
            virtual_deployment_detaails = deployment_details.get(u'entity').get(u'status').get(u'virtual_deployment_downloads')
            if virtual_deployment_detaails is not None:
                url = virtual_deployment_detaails[0].get(u'url')
            else:
                url = None
        except Exception as e:
            raise WMLClientError(u'Getting download url from deployment details failed.', e)

        if url is None:
            raise MissingValue(u'deployment_details.entity.virtual_deployment_downloads.url')

        return url

    @docstring_parameter({'str_type': STR_TYPE_NAME})
    def download(self, virtual_deployment_uid, filename=None):
        """
        Downloads file deployment of specified deployment Id. Currently supported format is Core ML.

        **Parameters**

        .. important::

            #. **virtual_deployment_uid**:  Unique Id of virtual deployment.\n
               **type**: str\n
            #. **filename**: filename of downloaded archive (optional).\n
               **type**: str\n

        **Output**

        .. important::
            **returns**: Path to downloaded file.\n
            **return type**: str\n

        **Example**

            >>> client.deployments.download(virtual_deployment_uid)
        """
        self._client._check_if_space_is_set()
        virtual_deployment_uid = str_type_conv(virtual_deployment_uid)
        Deployments._validate_type(virtual_deployment_uid, u'deployment_uid', STR_TYPE, False)

        if virtual_deployment_uid is not None and not is_uid(virtual_deployment_uid):
            raise WMLClientError(u'\'deployment_uid\' is not an uid: \'{}\''.format(virtual_deployment_uid))
        params = self._client._params()
        details = self.get_details(virtual_deployment_uid)
        download_url = self.get_download_url(details)
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35:
            download_url = download_url + "&version=2020-08-01"
            params = None
        if not self._ICP:
            # if self._client.CLOUD_PLATFORM_SPACES:
            #     download_url = download_url + "&version=2020-08-01"
            #     response = requests.get(
            #         download_url,
            #         headers=self._client._get_headers()
            #     )
            # else:
            response = requests.get(
                    download_url,
                    params=params,
                    headers=self._client._get_headers()
                )
        else:
            response = requests.get(
                download_url,
                params=params,
                headers=self._client._get_headers(),
                verify=False
            )
        if filename is None:
            filename = 'mlartifact.tar.gz'

        if response.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response.content)
                new_file.close()

                print_text_header_h2(
                    u'Successfully downloaded deployment file: ' + str(filename))

                return filename
        else:
            raise WMLClientError(u'Unable to download deployment content: ' + response.text)

    # @docstring_parameter({'str_type': STR_TYPE_NAME})
    # def score(self, deployment_id, meta_props, transaction_id=None):
    #         """
    #             Make scoring requests against deployed artifact.
    #
    #             **Parameters**
    #
    #             .. important::
    #                 #. **deployment_id**:  UID of the deployment to be scored\n
    #                    **type**: str\n
    #                 #. **meta_props**:  Meta props for scoring\n
    #                                     >>> Use client.deployments.ScoringMetaNames.show() to view the list of ScoringMetaNames.
    #                    **type**: dict\n
    #                 #. **transaction_id**:  transaction id to be passed with records during payload logging (optional)\n
    #                    **type**: str\n
    #
    #             **Output**
    #
    #             .. important::
    #                 **returns**: scoring result containing prediction and probability\n
    #                 **return type**: dict\n
    #
    #             .. note::
    #
    #                     * client.deployments.ScoringMetaNames.SCORING_INPUT_DATA is the only metaname valid for sync scoring.\n
    #
    #             **Example**
    #
    #              >>> scoring_payload = {wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{'fields': ['GENDER','AGE','MARITAL_STATUS','PROFESSION'], 'values': [['M',23,'Single','Student'],['M',55,'Single','Executive']]}]}
    #              >>> predictions = client.deployments.score(deployment_id, scoring_payload)
    #              >>> predictions = client.deployments.score(deployment_id, scoring_payload,async=True)
    #         """
    #         Deployments._validate_type(deployment_id, u'scoring_url', STR_TYPE, True)
    #         Deployments._validate_type(meta_props, u'meta_props', dict, True)
    #
    #
    #         score_payload = {}
    #         if meta_props[self.ScoringMetaNames.INPUT_DATA] is None:
    #             raise WMLClientError("Scoring data input is mandatory for synchronous scoring")
    #
    #         score_payload["input_data"] = meta_props[self.ScoringMetaNames.INPUT_DATA]
    #         payload = score_payload
    #         Deployments._validate_type(deployment_id, u'scoring_url', STR_TYPE, True)
    #         Deployments._validate_type(payload, u'payload', dict, True)
    #
    #         headers = self._client._get_headers()
    #
    #         if transaction_id is not None:
    #             headers.update({'x-global-transaction-id': transaction_id})
    #         # making change - connection keep alive
    #         scoring_url = self._wml_credentials["url"] + "/v4/deployments/"+deployment_id+"/predictions"
    #         if not self._ICP:
    #             response_scoring = self.session.post(
    #                 scoring_url,
    #                 json=payload,
    #                 headers=headers)
    #         else:
    #             response_scoring = self.session.post(
    #                 scoring_url,
    #                 json=payload,
    #                 headers=headers,
    #                 verify=False)
    #
    #         return self._handle_response(200, u'scoring', response_scoring)

    def list(self, limit=None):
        """
           List deployments. If limit is set to None there will be only first 50 records shown.
           
           **Parameters**
           
           .. important::
                #. **limit**:  limit number of fetched records\n
                   **type**: int\n
           
           **Output**
           
           .. important::
                This method only prints the list of all deployments in a table format.\n
                **return type**: None\n

           **Example**
           
            >>> client.deployments.list()

        """
        ##To be removed once deployments adds support for deployments
        self._client._check_if_space_is_set()

        details = self.get_details(limit=limit)
        resources = details[u'resources']

        values = []
        index = 0

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            # Ideally this needs to be on the server side
            if limit is not None and index == limit:
                break

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                if 'guid' in m[u'metadata']:
                    values.append((m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'status']['state'],
                                  m[u'metadata'][u'created_at'], self._get_deployable_asset_type(m)))
                else:
                    values.append((m[u'metadata'][u'id'], m[u'entity'][u'name'], m[u'entity'][u'status']['state'],
                                  m[u'metadata'][u'created_at'], self._get_deployable_asset_type(m)))
            else:
                if 'guid' in m[u'metadata']:
                    values.append((m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'status']['state'],
                                  m[u'metadata'][u'created_at']))
                else:
                    values.append((m[u'metadata'][u'id'], m[u'entity'][u'name'], m[u'entity'][u'status']['state'],
                                  m[u'metadata'][u'created_at']))

            index = index + 1

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            self._list(values, [u'GUID', u'NAME', u'STATE', u'CREATED', u'ARTIFACT_TYPE'], limit, 50)
        else:
            self._list(values, [u'GUID', u'NAME', u'STATE', u'CREATED'], limit, 50)

    def list_jobs(self,limit=None):
        """
        List the async deployment jobs. If limit is set to None there will be only first 50 records shown.

        **Parameters**

        .. important::
            #. **limit**:  limit number of fetched records\n
               **type**: int\n

        **Output**

        .. important::

            This method only prints the list of all async jobs in a table format.\n
                **return type**: None\n

        .. note::
            * This method list only async deployment jobs created for WML deployment.

        **Example**

         >>> client.deployments.list_jobs()
        """

        details = self.get_job_details(limit=limit)
        resources = details[u'resources']
        values = []
        index = 0

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            if limit is not None and index == limit:
                break

            if 'scoring' in m['entity']:
                state = m['entity']['scoring']['status']['state']
            else:
                state = m['entity']['decision_optimization']['status']['state']

            if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                deploy_id = m['entity']['deployment']['id']
                values.append((m[u'metadata'][u'id'], state, m[u'metadata'][u'created_at'], deploy_id))
            else:
                deploy_id = m['entity']['deployment']['href'].split("/")[3]
                values.append((m[u'metadata'][u'guid'], state, m[u'metadata'][u'created_at'], deploy_id))

            index = index + 1

        self._list(values, [u'JOB-UID', u'STATE', u'CREATED',u'DEPLOYMENT-ID'], limit, 50)

    def _get_deployable_asset_type(self, details):
        url = details[u'entity'][u'asset']['id']
        if 'model' in url:
            return 'model'
        elif 'function' in url:
            return 'function'
        else:
            return 'unknown'

    def update(self, deployment_uid, changes):
        """
                Updates existing deployment metadata. If ASSET is patched, then 'id' field is mandatory and
                it starts a deployment with the provided asset id/rev. Deployment id remains same
                
                **Parameters**
                
                .. important::

                    #. **deployment_uid**: Unique Id of deployment which  should be updated\n
                       **type**: str\n
                    #. **changes**:  elements which should be changed, where keys are ConfigurationMetaNames\n
                       **type**: dict\n

                **Output**
                
                .. important::

                    **returns**: metadata of updated deployment\n
                    **return type**: dict\n
                    
                **Example**
                
                 >>> metadata = {
                 >>> client.deployments.ConfigurationMetaNames.NAME:"updated_Deployment",
                 >>> client.deployments.ConfigurationMetaNames.ASSET: { "id": "ca0cd864-4582-4732-b365-3165598dc945", "rev":"2" }
                 >>> }
                 >>>
                 >>> deployment_details = client.deployments.update(deployment_uid, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##To be removed once deployments adds support for deployments
        self._client._check_if_space_is_set()

        ret202 = False

        ## In case of passing 'AUTO_ONLINE_DEPLOYMENT' as true, we need to poll for deployment to be either 'deploy_success' or 'update_success'.

        deployment_uid = str_type_conv(deployment_uid)
        Deployments._validate_type(deployment_uid, 'deployment_uid', STR_TYPE, True)

        if 'asset' in changes and not changes[u'asset']:
            msg = "ASSETS cannot be empty. 'id' and 'rev' fields are supported. 'id' is mandatory"
            print(msg)
            raise WMLClientError(msg)

        # if changes.get('asset') is not None and (changes.get('name') is not None or changes.get('description') is not None):
        if changes.get('asset') is not None and (len(changes) > 1):
            msg = "When ASSET is being updated/patched, other fields cannot be updated. If other fields are to be " \
                  "updated, try without ASSET update. ASSET update triggers deployment with the new asset retaining " \
                  "the same deployment_id"
            print(msg)
            raise WMLClientError(msg)

        deployment_details = self.get_details(deployment_uid)
        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(deployment_details['entity'], changes,with_validation=True)

        ## As auto_online_deployment and auto_redeploy values are passed as 'bool' but service needs them in 'str' format to patch.
        for ele in patch_payload:
            if('auto_online_deployment' in ele['path'] or 'auto_redeploy' in ele['path']):
                if(ele['value']==True):
                    ele['value']='true'
                else:
                    ele['value']='false'

        url = self._href_definitions.get_deployment_href(deployment_uid)

        if not self._ICP:
            response = requests.patch(url, json=patch_payload, params=self._client._params(),headers=self._client._get_headers())
        else:
            response = requests.patch(url, json=patch_payload, params=self._client._params(),headers=self._client._get_headers(), verify=False)

        if 'asset' in changes and response.status_code == 202:
            updated_details = self._handle_response(202, u'deployment asset patch', response)

            ret202 = True

            print("Since ASSET is patched, deployment with new asset id/rev is being started. " \
                  "Monitor the status using deployments.get_details(deployment_uid) api")
        elif 'hardware_spec' in changes:
            updated_details = self._handle_response(202, u'deployment scaling', response)
            ret202 = True
        else:
            updated_details = self._handle_response(200, u'deployment patch', response)

        if('auto_online_deployment' in changes):
           if response is not None:
               if response.status_code == 200:
                   deployment_details = self.get_details(deployment_uid)

                   import time
                   print_text_header_h1(u' deployment update for uid: \'{}\' started'.format(deployment_uid))

                   status = deployment_details[u'entity'][u'status'][u'state']

                   with StatusLogger(status) as status_logger:
                       while True:
                           time.sleep(5)
                           deployment_details = self.get_details(deployment_uid)
                           status = deployment_details[u'entity'][u'status'][u'state']
                           status_logger.log_state(status)

                           if status != u'DEPLOY_IN_PROGRESS' and status != u'UPDATE_IN_PROGRESS':
                               break

                   if status == u'DEPLOY_SUCCESS' or status == u'UPDATE_SUCCESS':
                       print(u'')
                       print_text_header_h2(
                           u'Successfully finished deployment update, deployment_uid=\'{}\''.format(deployment_uid))
                       return deployment_details
                   else:
                       print_text_header_h2(u'Deployment update failed')
                       self._deployment_status_errors_handling(deployment_details, 'update', deployment_uid)
               else:
                   error_msg = u'Deployment update failed'
                   reason = response.text
                   print(reason)
                   print_text_header_h2(error_msg)
                   raise WMLClientError(error_msg + u'. Error: ' + str(response.status_code) + '. ' + reason)

        if not ret202:
            return updated_details

    ## Below functions are for async scoring. They are just dummy functions.
    def _score_async(self,deployment_uid, scoring_payload, transaction_id=None):

        Deployments._validate_type(deployment_uid, u'deployment_id', STR_TYPE, True)
        Deployments._validate_type(scoring_payload, u'scoring_payload', dict, True)
        headers = self._client._get_headers()

        if transaction_id is not None:
            headers.update({'x-global-transaction-id': transaction_id})
        # making change - connection keep alive
        scoring_url = self._href_definitions.get_async_deployment_job_href()
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
        else:
            params = None

        if not self._ICP:
            response_scoring = self.session.post(
                scoring_url,
                params=params,
                json=scoring_payload,
                headers=headers)
        else:
            response_scoring = self.session.post(
                scoring_url,
                params = params,
                json=scoring_payload,
                headers=headers,
                verify=False)

        return self._handle_response(202, u'scoring asynchronously', response_scoring)

    def create_job(self, deployment_id, meta_props):
        """
            Create an asynchronous deployment job.

            **Parameters**

            .. important::

                #. **deployment_id**:  Unique Id of Deployment\n
                   **type**: str\n

                #. **meta_props**: metaprops. To see the available list of metanames use:\n
                                   >>> client.deployments.ScoringMetaNames.get() or client.deployments.DecisionOptimizationmetaNames.get()

                   **type**: dict\n

            **Output**

            .. important::

                **returns**: metadata of the created async deployment job\n
                **return type**: dict\n

            .. note::

                * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

            **Example**

             >>> scoring_payload = {wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{'fields': ['GENDER','AGE','MARITAL_STATUS','PROFESSION'], 'values': [['M',23,'Single','Student'],['M',55,'Single','Executive']]}]}
             >>> async_job = client.deployments.create_job(deployment_id, scoring_payload)

         """

        WMLResource._chk_and_block_create_update_for_python36(self)
        Deployments._validate_type(deployment_id, u'deployment_uid', STR_TYPE, True)
        Deployments._validate_type(meta_props, u'meta_props', dict, True)

        scoring_data = None
        flag = 0 ## To see if it is async scoring or DecisionOptimization Job
        deployment_details = self.get_details(deployment_id)

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            asset = deployment_details["entity"]["asset"]['id']
        else:
            asset = deployment_details["entity"]["asset"]["href"].split("/")[-1]

        do_model = False
        if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            asset_details = self._client.data_assets.get_details(asset)
            if 'wml_model' in asset_details["entity"] and 'type' in asset_details['entity']['wml_model']:
                if 'do' in asset_details['entity']['wml_model']['type']:
                    do_model = True
        else:
            asset_details = self._client.repository.get_details(asset)
            if "type" in asset_details["entity"]:
                if "do" in asset_details["entity"]["type"]:
                    do_model = True

        if do_model:
            payload = self.DecisionOptimizationMetaNames._generate_resource_metadata(meta_props,
                                                                                     with_validation=True,
                                                                                     client=self._client)
            flag = 1
        else:
            payload = self.ScoringMetaNames._generate_resource_metadata(meta_props, with_validation=True,
                                                                        client=self._client)

        if "scoring" in payload and "input_data" in payload["scoring"]:
                scoring_data = payload["scoring"]["input_data"]

        if "decision_optimization" in payload and "input_data" in payload["decision_optimization"]:
                scoring_data = payload["decision_optimization"]["input_data"]

        if scoring_data is not None:
            score_payload = []
            for each_score_request in scoring_data:
                lib_checker.check_lib(lib_name='pandas')
                import pandas as pd
                if "values" in each_score_request:
                    scoring_values = each_score_request["values"]
                    # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
                    if (isinstance(scoring_values, pd.DataFrame)):
                        fields_names = scoring_values.columns.values.tolist()
                        values = scoring_values.values.tolist()

                        each_score_request["values"] = values
                        if fields_names is not None:
                            each_score_request["fields"] = fields_names


                    ## If payload is a numpy dataframe

                    elif (isinstance(scoring_values, np.ndarray)):

                        values = scoring_values.tolist()
                        each_score_request["values"] = values


                # else:
                #     payload_score["values"] = each_score_request["values"]
                #     if "fields" in each_score_request:
                #         payload_score["fields"] = each_score_request["fields"]
                score_payload.append(each_score_request)

            ##See if it is scoring or DecisionOptimizationJob

            if flag == 0:
                payload["scoring"]["input_data"] = score_payload
            if flag == 1:
                payload["decision_optimization"]["input_data"] = score_payload

        if self._client.ICP_30 or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            import copy
            if 'input_data_references' in meta_props:
                Deployments._validate_type(meta_props.get('input_data_references'), u'input_data_references', list,
                                           True)
                modified_input_data_references=False
                input_data = copy.deepcopy(meta_props.get('input_data_references'))
                for i, input_data_fields in enumerate(input_data):
                    if 'connection' not in input_data_fields:
                        modified_input_data_references=True
                        input_data_fields.update({'connection': {}})
                if modified_input_data_references:
                    if 'scoring' in payload:
                        payload['scoring'].update({'input_data_references': input_data})
                    else:
                        payload['decision_optimization'].update({'input_data_references': input_data})

            if 'output_data_reference' in meta_props:
                Deployments._validate_type(meta_props.get('output_data_reference'), u'output_data_reference', dict,
                                           True)

                output_data = copy.deepcopy(meta_props.get('output_data_reference'))
                if 'connection' not in output_data:  # and output_data.get('connection', None) is not None:
                    output_data.update({'connection': {}})
                    payload['scoring'].update({'output_data_reference': output_data})

            if 'output_data_references' in meta_props:
                Deployments._validate_type(meta_props.get('output_data_references'), u'output_data_references', list, True)
                output_data = copy.deepcopy(meta_props.get('output_data_references'))
                modified_output_data_references = False
                for i, output_data_fields in enumerate(output_data):
                    if 'connection' not in output_data_fields:
                        modified_output_data_references = True
                        output_data_fields.update({'connection': {}})
                if modified_output_data_references and 'decision_optimization' in payload:
                    payload['decision_optimization'].update({'output_data_references': output_data})

            payload.update({"deployment": {"id": deployment_id}})
            if 'hardware_spec' in meta_props:
                payload.update({"hardware_spec": meta_props[self.ConfigurationMetaNames.HARDWARE_SPEC]})
            if 'hybrid_pipeline_hardware_specs' in meta_props:
                payload.update({"hybrid_pipeline_hardware_specs": meta_props[self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS]})

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                payload.update({'space_id': self._client.default_space_id})

                if 'name' not in payload:
                    import uuid
                    payload.update({'name': 'name_' + str(uuid.uuid4())})
        else:
            payload.update({"deployment": {"href": "/v4/deployments/" + deployment_id}})

        return self._score_async(deployment_id, payload, transaction_id=None)

    def get_job_details(self, job_uid=None, limit=None):
        """
           Get information about your deployment job(s). If  deployment job_uid is not passed, all deployment jobs details are fetched.

           **Parameters**

           .. important::

                #. **job_uid**: Unique Job ID (optional)\n
                   **type**: str\n
                #. **limit**:  limit number of fetched records (optional)\n
                   **type**: int\n

           **Output**

           .. important::
                **returns**: metadata of deployment job(s)\n
                **return type**: dict\n
                dict (if job_uid is not None) or {"resources": [dict]} (if job_uid is None)\n

           .. note::
                If job_uid is not specified, all deployment jobs metadata associated with the deployment Id is fetched\n


           **Example**

            >>> deployment_details = client.deployments.get_job_details()
            >>> deployments_details = client.deployments.get_job_details(job_uid=job_uid)

        """
        if job_uid is not None:
            Deployments._validate_type(job_uid, u'job_uid', STR_TYPE, True)
        url = self._href_definitions.get_async_deployment_job_href()

        if self._ICP:
            if job_uid is not None:
                url = url + '/' + job_uid
                response = requests.get(
                    url,
                    headers=self._client._get_headers(),
                    params=self._client._params(),
                    verify=False
                )
                return self._handle_response(200, "async deployment job", response)
            else:
                return self._get_artifact_details(url, job_uid, limit, 'async deployment jobs')
        else:
            return self._get_artifact_details(url, job_uid, limit, 'async deployment jobs')

    def get_job_status(self, job_id):
        """
           Get the status of the deployment job.

           **Parameters**

           .. important::
                #. **job_id**: Unique Id of the deployment job\n
                   **type**: str\n

           **Output**

           .. important::
                **returns**: status of the deployment job\n
                **return type**: dict\n

           **Example**

            >>> job_status = client.deployments.get_job_status(job_uid)

        """

        job_details = self.get_job_details(job_id)

        if 'scoring' not in job_details['entity']:
            return job_details['entity']['decision_optimization']['status']
        return job_details['entity']['scoring']['status']

    def get_job_uid(self, job_details):
        """
           Get the Unique Id of the deployment job.

           **Parameters**

           .. important::
                #. **job_details**: metadata of the deployment job\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: Unique Id of the deployment job\n
                **return type**: str\n

           **Example**

            >>> job_details = client.deployments.get_job_details(job_uid=job_uid)
            >>> job_status = client.deployments.get_job_uid(job_details)

        """

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            return job_details['metadata']['id']
        else:
            return job_details['metadata']['guid']

    def get_job_href(self, job_details):
        """
           Get the href of the deployment job.

           **Parameters**

           .. important::
                #. **job_details**: metadata of the deployment job\n
                   **type**: dict\n

           **Output**

           .. important::
                **returns**: href of the deployment job\n
                **return type**: str\n

           **Example**

            >>> job_details = client.deployments.get_job_details(job_uid=job_uid)
            >>> job_status = client.deployments.get_job_href(job_details)

        """

        if self._client.ICP_30 and not self._client.ICP_PLATFORM_SPACES:
            job_href = "/v4/deployment_jobs/" + job_details['metadata']['id']
            return job_href
        else:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                return "/ml/v4/deployment_jobs/{}".format(job_details[u'metadata'][u'id'])
            else:
                return job_details['metadata']['href']


    def delete_job(self, job_uid, hard_delete=False):
        """
        Cancels a deployment job that is currenlty running.  This method is also be used to delete metadata
        details of the completed or canceled jobs when hard_delete parameter is set to True.

        **Parameters**

        .. important::

            #. **job_uid**:  Unique Id of deployment job which  should be canceled\n
               **type**: str\n

            #. **hard_delete**: specify True or False.
                                True - To delete the completed or canceled job.
                                False - To cancel the currently running deployment job. Default value is False.
               **type**: Boolean\n


        **Output**

        .. important::

            **returns**: status ("SUCCESS" or "FAILED")\n
            **return type**: str\n

        **Example**

         >>> client.deployments.delete_job(job_uid)

        """

        job_uid = str_type_conv(job_uid)

        Deployments._validate_type(job_uid, u'job_uid', STR_TYPE, True)

        if job_uid is not None and not is_uid(job_uid):
            raise WMLClientError(u'\'job_uid\' is not an uid: \'{}\''.format(job_uid))

        url = self._href_definitions.get_async_deployment_jobs_href(job_uid)

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
        else:
            params = {}

        if hard_delete is True:
            params.update({'hard_delete': u'true'})

        if not self._ICP:
            response_delete = requests.delete(
                url,
                headers=self._client._get_headers(),
                params=params)
        else:
            response_delete = requests.delete(
                url,
                headers=self._client._get_headers(),
                params=params,
                verify=False)

        return self._handle_response(204, u'deployment async job deletion', response_delete, False)


    # def create_environment(self, meta_props):
    #     """
    #             Cancels a job.
    #
    #             **Parameters**
    #
    #             .. important::
    #
    #                 #. **job_uid**:  UID of job which  should be canceled\n
    #                    **type**: str\n
    #
    #             **Output**
    #
    #             .. important::
    #
    #                 **returns**: status ("SUCCESS" or "FAILED")\n
    #                 **return type**: str\n
    #
    #             **Example**
    #
    #              >>> client.deployments.delete_job(job_uid)
    #     """
    #
    #     Deployments._validate_type(meta_props, u'metaprops', dict, True)
    #
    #
    #     url = self._href_definitions.get_envs_href()
    #     metadata = self.EnvironmentMetanames._generate_resource_metadata(meta_props)
    #
    #     if not self._ICP:
    #         response_post = requests.post(url, json=metadata,
    #                                           headers=self._client._get_headers())
    #
    #
    #     else:
    #         response_post = requests.post(url, json=metadata,
    #                                       headers=self._client._get_headers(), verify=False)
    #
    #     return self._handle_response(201, u'deployment job deletion', response_post, False)
    #
    # def delete_environment(self, env_id):
    #     """
    #             Cancels a job.
    #
    #             **Parameters**
    #
    #             .. important::
    #
    #                 #. **job_uid**:  UID of job which  should be canceled\n
    #                    **type**: str\n
    #
    #             **Output**
    #
    #             .. important::
    #
    #                 **returns**: status ("SUCCESS" or "FAILED")\n
    #                 **return type**: str\n
    #
    #             **Example**
    #
    #              >>> client.deployments.delete_job(job_uid)
    #     """
    #
    #     env_id = str_type_conv(env_id)
    #
    #     Deployments._validate_type(env_id, u'env_uid', STR_TYPE, True)
    #
    #
    #     if env_id is not None and not is_uid(env_id):
    #         raise WMLClientError(u'\'env_id\' is not an uid: \'{}\''.format(env_id))
    #
    #
    #     url = self._href_definitions.get_env_href(env_id)
    #
    #     if not self._ICP:
    #         response_delete = requests.delete(
    #             url,
    #             headers=self._client._get_headers())
    #     else:
    #         response_delete = requests.delete(
    #             url,
    #             headers=self._client._get_headers(),
    #             verify=False)
    #
    #     return self._handle_response(204, u'environment deletion', response_delete, False)
    #
    # def get_environment_uid(self, env_details):
    #     """
    #        Get the UID of the environment.
    #
    #        **Parameters**
    #
    #        .. important::
    #             #. **job_details**: metadata of the job\n
    #                **type**: dict\n
    #
    #        **Output**
    #
    #        .. important::
    #             **returns**: UID of the job\n
    #             **return type**: str\n
    #
    #        **Example**
    #
    #         >>> job_details = client.deployments.get_job_details(deployment_uid=deployment_uid,job_uid=job_uid)
    #         >>> job_status = client.deployments.get_job_uid(job_details)
    #     """
    #
    #     return env_details['metadata']['guid']
    #
    # def get_environment_href(self, env_details):
    #     """
    #        Get the href of the job.
    #
    #        **Parameters**
    #
    #        .. important::
    #             #. **job_details**: metadata of the job\n
    #                **type**: dict\n
    #
    #        **Output**
    #
    #        .. important::
    #             **returns**: href of the job\n
    #             **return type**: str\n
    #
    #        **Example**
    #
    #         >>> job_details = client.deployments.get_job_details(deployment_uid=deployment_uid,job_uid=job_uid)
    #         >>> job_status = client.deployments.get_job_href(job_details)
    #     """
    #
    #     return env_details['metadata']['href']
    #
    # def get_environment_details(self,env_id=None,limit=None):
    #     """
    #        Get information about your job(s). If job_uid is not passed, all jobs details are fetched.
    #
    #        **Parameters**
    #
    #        .. important::
    #
    #             #. **job_uid**: Job UID (optional)\n
    #                **type**: str\n
    #             #. **limit**:  limit number of fetched records (optional)\n
    #                **type**: int\n
    #
    #        **Output**
    #
    #        .. important::
    #             **returns**: metadata of job(s)\n
    #             **return type**: dict\n
    #             dict (if job UID is not None) or {"resources": [dict]} (if job UID is None)\n
    #
    #        .. note::
    #             If job UID is not specified, all jobs metadata associated with the deployment UID is fetched\n
    #
    #
    #        **Example**
    #
    #         >>> deployment_details = client.deployments.get_job_details()
    #         >>> deployments_details = client.deployments.get_job_details(job_uid=job_uid)
    #     """
    #     #Deployments._validate_type(job_uid, u'deployment_uid', STR_TYPE, True)
    #     #Deployments._validate_type(job_uid, u'job_uid', STR_TYPE, True)
    #     url = self._href_definitions.get_envs_href()
    #     return self._get_artifact_details(url, env_id, limit, 'get environments details')
    #
    # def list_environments(self,limit=None):
    #     """
    #        List the async jobs. If limit is set to None there will be only first 50 records shown.
    #
    #        **Parameters**
    #
    #        .. important::
    #             #. **limit**:  limit number of fetched records\n
    #                **type**: int\n
    #
    #        **Output**
    #
    #        .. important::
    #             This method only prints the list of all async jobs in a table format.\n
    #             **return type**: None\n
    #
    #        **Example**
    #
    #         >>> client.deployments.list_jobs()
    #     """
    #     details = self.get_environment_details()
    #     resources = details[u'resources']
    #     values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'],
    #                m[u'metadata'][u'created_at']) for m in resources]
    #
    #     self._list(values, [u'GUID', u'NAME', u'CREATED'], None, 50)

