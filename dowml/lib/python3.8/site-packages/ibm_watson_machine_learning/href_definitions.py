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

import re

TRAINING_MODEL_HREF_PATTERN = u'{}/v4/trainings/{}'
TRAINING_MODELS_HREF_PATTERN = u'{}/v4/trainings'
REPO_MODELS_FRAMEWORKS_HREF_PATTERN = u'{}/v3/models/frameworks'

INSTANCE_ENDPOINT_HREF_PATTERN = u'{}/v3/wml_instance'
INSTANCE_BY_ID_ENDPOINT_HREF_PATTERN = u'{}/v3/wml_instances/{}'
TOKEN_ENDPOINT_HREF_PATTERN = u'{}/v3/identity/token'
EXPERIMENTS_HREF_PATTERN = u'{}/v4/experiments'
EXPERIMENT_HREF_PATTERN = u'{}/v4/experiments/{}'
EXPERIMENT_RUNS_HREF_PATTERN = u'{}/v3/experiments/{}/runs'
EXPERIMENT_RUN_HREF_PATTERN = u'{}/v3/experiments/{}/runs/{}'

PUBLISHED_MODEL_HREF_PATTERN = u'{}/v4/models/{}'
PUBLISHED_MODELS_HREF_PATTERN = u'{}/v4/models'
LEARNING_CONFIGURATION_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/learning_configuration'
LEARNING_ITERATION_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/learning_iterations/{}'
LEARNING_ITERATIONS_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/learning_iterations'
EVALUATION_METRICS_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/evaluation_metrics'
FEEDBACK_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/feedback'

DEPLOYMENTS_HREF_PATTERN = u'{}/v4/deployments'
DEPLOYMENT_HREF_PATTERN = u'{}/v4/deployments/{}'
DEPLOYMENT_JOB_HREF_PATTERN = u'{}/v4/deployment_jobs'
DEPLOYMENT_JOBS_HREF_PATTERN = u'{}/v4/deployment_jobs/{}'
DEPLOYMENT_ENVS_HREF_PATTERN = u'{}/v4/deployments/environments'
DEPLOYMENT_ENV_HREF_PATTERN = u'{}/v4/deployments/environments/{}'

MODEL_LAST_VERSION_HREF_PATTERN = u'{}/v4/models/{}'
DEFINITION_HREF_PATTERN = u'{}/v3/ml_assets/training_definitions/{}'
DEFINITIONS_HREF_PATTERN = u'{}/v3/ml_assets/training_definitions'

FUNCTION_HREF_PATTERN = u'{}/v4/functions/{}'
FUNCTION_LATEST_CONTENT_HREF_PATTERN = u'{}/v4/functions/{}/content'
FUNCTIONS_HREF_PATTERN = u'{}/v4/functions'

RUNTIME_HREF_PATTERN = u'{}/v4/runtimes/{}'
RUNTIMES_HREF_PATTERN = u'{}/v4/runtimes'
CUSTOM_LIB_HREF_PATTERN = u'{}/v4/libraries/{}'
CUSTOM_LIBS_HREF_PATTERN = u'{}/v4/libraries'

IAM_TOKEN_API = u'{}&grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey'
IAM_TOKEN_URL = u'{}/oidc/token'
PROD_SVT_URL = ['https://us-south.ml.cloud.ibm.com',
                'https://eu-gb.ml.cloud.ibm.com',
                'https://eu-de.ml.cloud.ibm.com',
                'https://jp-tok.ml.cloud.ibm.com',
                'https://ibm-watson-ml.mybluemix.net',
                'https://ibm-watson-ml.eu-gb.bluemix.net',
                'https://private.us-south.ml.cloud.ibm.com',
                'https://private.eu-gb.ml.cloud.ibm.com',
                'https://private.eu-de.ml.cloud.ibm.com',
                'https://private.jp-tok.ml.cloud.ibm.com',
                'https://yp-qa.ml.cloud.ibm.com',
                'https://private.yp-qa.ml.cloud.ibm.com',
                'https://yp-cr.ml.cloud.ibm.com',
                'https://private.yp-cr.ml.cloud.ibm.com']

PIPELINES_HREF_PATTERN=u'{}/v4/pipelines'
PIPELINE_HREF_PATTERN=u'{}/v4/pipelines/{}'


SPACES_HREF_PATTERN = u'{}/v4/spaces'
SPACE_HREF_PATTERN = u'{}/v4/spaces/{}'
MEMBER_HREF_PATTERN=u'{}/v4/spaces/{}/members/{}'
MEMBERS_HREF_PATTERN=u'{}/v4/spaces/{}/members'

SPACES_PLATFORM_HREF_PATTERN = u'{}/v2/spaces'
SPACE_PLATFORM_HREF_PATTERN = u'{}/v2/spaces/{}'
SPACES_MEMBERS_HREF_PATTERN = u'{}/v2/spaces/{}/members'
SPACES_MEMBER_HREF_PATTERN = u'{}/v2/spaces/{}/members/{}'

V4_INSTANCE_ID_HREF_PATTERN = u'{}/ml/v4/instances/{}'

API_VERSION = u'/v4'
SPACES=u'/spaces'
PIPELINES=u'/pipelines'
EXPERIMENTS=u'/experiments'
LIBRARIES=u'/libraries'
RUNTIMES=u'/runtimes'
SOFTWARE_SPEC=u'/software_specifications'
DEPLOYMENTS = u'/deployments'
MODEL_DEFINITION_ASSETS = u'{}/v2/assets'
MODEL_DEFINITION_SEARCH_ASSETS = u'{}/v2/asset_types/wml_model_definition/search'
WSD_SEARCH_ASSETS = u'{}/v2/asset_types/{}/search'
DATA_ASSETS = u'{}/v2/assets'
DATA_ASSET = u'{}/v2/assets/{}'
ASSET = u'{}/v2/assets/{}'
ASSETS = u'{}/v2/assets'
ATTACHMENT = u'{}/v2/assets/{}/attachments/{}'
ATTACHMENT_COMPLETE = u'{}/v2/assets/{}/attachments/{}/complete'
ATTACHMENTS = u'{}/v2/assets/{}/attachments'
SEARCH_ASSETS = u'{}/v2/asset_types/data_asset/search'
SEARCH_SHINY = u'{}/v2/asset_types/shiny_asset/search'
SEARCH_SCRIPT = u'{}/v2/asset_types/script/search'
ASSET_FILES = u'{}/v2/asset_files/'
ASSET_TYPE = u'{}/v2/asset_types'
DATA_SOURCE_TYPE = u'{}/v2/datasource_types'
DATA_SOURCE_TYPE_BY_ID = u'{}/v2/datasource_types/{}'
CONNECTION_ASSET = u'{}/v2/connections'
CONNECTION_ASSET_SEARCH = u'{}/v2/connections'
CONNECTION_BY_ID = u'{}/v2/connections/{}'
SOFTWARE_SPECIFICATION = u'{}/v2/software_specifications/{}'
SOFTWARE_SPECIFICATIONS = u'{}/v2/software_specifications'
HARDWARE_SPECIFICATION = u'{}/v2/hardware_specifications/{}'
HARDWARE_SPECIFICATIONS = u'{}/v2/hardware_specifications'
PACKAGE_EXTENSION = u'{}/v2/package_extensions/{}'
PACKAGE_EXTENSIONS = u'{}/v2/package_extensions'
PROJECT = u'{}/v2/projects/{}'

V4GA_CLOUD_MIGRATION = u'{}/ml/v4/repository'
V4GA_CLOUD_MIGRATION_ID = u'{}/ml/v4/repository/{}'

REMOTE_TRAINING_SYSTEM = u'{}/v4/remote_training_systems'
REMOTE_TRAINING_SYSTEM_ID = u'{}/v4/remote_training_systems/{}'

EXPORTS = u'{}/v2/asset_exports'
EXPORT_ID = u'{}/v2/asset_exports/{}'
EXPORT_ID_CONTENT = u'{}/v2/asset_exports/{}/content'

IMPORTS = u'{}/v2/asset_imports'
IMPORT_ID = u'{}/v2/asset_imports/{}'

VOLUMES = u'{}/zen-data/v3/service_instances'
VOLUME_ID = u'{}/zen-data/v3/service_instances/{}'
VOLUME_SERVICE = u'{}/zen-data/v1/volumes/volume_services/{}'
VOLUME_SERVICE_FILE_UPLOAD = u'{}/zen-volumes/{}/v1/volumes/files/'

def is_url(s):
    res = re.match('https?:\/\/.+', s)
    return res is not None


def is_uid(s):
    res = re.match('[a-z0-9\-]{36}', s)
    return res is not None


class HrefDefinitions:
    def __init__(self, wml_credentials, cloud_platform_spaces=False, platform_url=None, cams_url=None, cp4d_platform_spaces=False):
        self._wml_credentials = wml_credentials
        self.cloud_platform_spaces = cloud_platform_spaces
        self.cp4d_platform_spaces = cp4d_platform_spaces
        self.platform_url = platform_url
        self.cams_url = cams_url

        if self.cloud_platform_spaces or self.cp4d_platform_spaces:
            self.prepend = '/ml'
        else:
            self.prepend = ''

    def get_training_href(self, model_uid):

        return TRAINING_MODEL_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, model_uid)

    def get_trainings_href(self):
        return TRAINING_MODELS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_repo_models_frameworks_href(self):
        return REPO_MODELS_FRAMEWORKS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_instance_endpoint_href(self):
        return INSTANCE_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_instance_by_id_endpoint_href(self):
        return INSTANCE_BY_ID_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'])

    def get_token_endpoint_href(self):
        return TOKEN_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_published_model_href(self, model_uid):
        return PUBLISHED_MODEL_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, model_uid)

    def get_published_models_href(self):
        return PUBLISHED_MODELS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_learning_configuration_href(self, model_uid):
        return LEARNING_CONFIGURATION_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_learning_iterations_href(self, model_uid):
        return LEARNING_ITERATIONS_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_learning_iteration_href(self, model_uid, iteration_uid):
        return LEARNING_ITERATION_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid, iteration_uid)

    def get_evaluation_metrics_href(self, model_uid):
        return EVALUATION_METRICS_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_feedback_href(self, model_uid):
        return FEEDBACK_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_model_last_version_href(self, artifact_uid):
        return MODEL_LAST_VERSION_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, artifact_uid)

    def get_deployments_href(self):
        return DEPLOYMENTS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_experiments_href(self):
        return EXPERIMENTS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_experiment_href(self, experiment_uid):
        return EXPERIMENT_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, experiment_uid)

    def get_experiment_runs_href(self, experiment_uid):
        return EXPERIMENT_RUNS_HREF_PATTERN.format(self._wml_credentials['url'], experiment_uid)

    def get_experiment_run_href(self, experiment_uid, experiment_run_uid):
        return EXPERIMENT_RUN_HREF_PATTERN.format(self._wml_credentials['url'], experiment_uid, experiment_run_uid)

    def get_deployment_href(self, deployment_uid):
        return DEPLOYMENT_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, deployment_uid)

    def get_definition_href(self, definition_uid):
        return DEFINITION_HREF_PATTERN.format(self._wml_credentials['url'], definition_uid)

    def get_definitions_href(self):
        return DEFINITIONS_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_function_href(self, ai_function_uid):
        return FUNCTION_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, ai_function_uid)

    def get_function_latest_revision_content_href(self, ai_function_uid):
        return FUNCTION_LATEST_CONTENT_HREF_PATTERN.format(self._wml_credentials['url'], ai_function_uid)

    def get_functions_href(self):
        return FUNCTIONS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_runtime_href_v4(self, runtime_uid):
        return u'/v4/runtimes/{}'.format(runtime_uid)

    def get_runtime_href(self, runtime_uid):
        return RUNTIME_HREF_PATTERN.format(self._wml_credentials['url'], runtime_uid)

    def get_runtimes_href(self):
        return RUNTIMES_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_custom_library_href(self, library_uid):
        return CUSTOM_LIB_HREF_PATTERN.format(self._wml_credentials['url'], library_uid)

    def get_custom_libraries_href(self):
        return CUSTOM_LIBS_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_pipeline_href(self, pipeline_uid):
        return PIPELINE_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, pipeline_uid)

    def get_pipelines_href(self):
        return PIPELINES_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_space_href(self, spaces_uid):
        return SPACE_HREF_PATTERN.format(self._wml_credentials['url'], spaces_uid)

    def get_spaces_href(self):
        return SPACES_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_platform_space_href(self, spaces_id):
        # return SPACE_PLATFORM_HREF_PATTERN.format(self._wml_credentials['url'], spaces_id)
        if self.platform_url is None:
            return SPACE_PLATFORM_HREF_PATTERN.format(self._wml_credentials['url'], spaces_id)
        else:
            return SPACE_PLATFORM_HREF_PATTERN.format(self.platform_url, spaces_id)

    def get_platform_spaces_href(self):
        if self.platform_url is None:
            return SPACES_PLATFORM_HREF_PATTERN.format(self._wml_credentials['url'])
        else:
            return SPACES_PLATFORM_HREF_PATTERN.format(self.platform_url)

    def get_platform_spaces_member_href(self, spaces_id, member_id):
        if self.platform_url is None:
            return SPACES_MEMBER_HREF_PATTERN.format(self._wml_credentials['url'], spaces_id, member_id)
        else:
            return SPACES_MEMBER_HREF_PATTERN.format(self.platform_url, spaces_id, member_id)

    def get_platform_spaces_members_href(self,spaces_id):
        if self.platform_url is None:
            return SPACES_MEMBERS_HREF_PATTERN.format(self._wml_credentials['url'], spaces_id)
        else:
            return SPACES_MEMBERS_HREF_PATTERN.format(self.platform_url, spaces_id)

    def get_v4_instance_id_href(self):
        return V4_INSTANCE_ID_HREF_PATTERN.format(self._wml_credentials['url'],
                                                  self._wml_credentials['instance_id'])

    def get_async_deployment_job_href(self):
        return DEPLOYMENT_JOB_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_async_deployment_jobs_href(self, job_uid):
        return DEPLOYMENT_JOBS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, job_uid)

    def get_iam_token_api(self):
        return IAM_TOKEN_API.format(self._wml_credentials['apikey'])

    def get_iam_token_url(self):
        if (self._wml_credentials['url'] in PROD_SVT_URL):
            return IAM_TOKEN_URL.format('https://iam.cloud.ibm.com')
        else:
            return IAM_TOKEN_URL.format('https://iam.test.cloud.ibm.com')
    def get_member_href(self, spaces_uid,member_id):
        return MEMBER_HREF_PATTERN.format(self._wml_credentials['url'], spaces_uid,member_id)

    def get_members_href(self,spaces_uid):
        return MEMBERS_HREF_PATTERN.format(self._wml_credentials['url'],spaces_uid)

    def get_data_asset_href(self,asset_id):
        # return DATA_ASSET.format(self._wml_credentials['url'],asset_id)
        if self.cams_url is None:
            return DATA_ASSET.format(self._wml_credentials['url'], asset_id)
        else:
            return DATA_ASSET.format(self.cams_url, asset_id)

    def get_data_assets_href(self):
        # return DATA_ASSETS.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return DATA_ASSETS.format(self._wml_credentials['url'])
        else:
            return DATA_ASSETS.format(self.cams_url)

    def get_assets_href(self):
        # return DATA_ASSETS.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return ASSETS.format(self._wml_credentials['url'])
        else:
            return ASSETS.format(self.cams_url)

    def get_asset_href(self,asset_id):
        # return ASSET.format(self._wml_credentials['url'],asset_id)
        if self.cams_url is None:
            return ASSET.format(self._wml_credentials['url'], asset_id)
        else:
            return ASSET.format(self.cams_url, asset_id)

    def get_attachment_href(self,asset_id,attachment_id):
        # return ATTACHMENT.format(self._wml_credentials['url'],asset_id,attachment_id)
        if self.cams_url is None:
            return ATTACHMENT.format(self._wml_credentials['url'], asset_id, attachment_id)
        else:
            return ATTACHMENT.format(self.cams_url, asset_id, attachment_id)

    def get_attachments_href(self, asset_id):
        # return ATTACHMENT.format(self._wml_credentials['url'],asset_id,attachment_id)
        if self.cams_url is None:
            return ATTACHMENTS.format(self._wml_credentials['url'], asset_id )
        else:
            return ATTACHMENTS.format(self.cams_url, asset_id)

    def get_attachment_complete_href(self,asset_id,attachment_id):
        # return ATTACHMENT.format(self._wml_credentials['url'],asset_id,attachment_id)
        if self.cams_url is None:
            return ATTACHMENT_COMPLETE.format(self._wml_credentials['url'], asset_id, attachment_id)
        else:
            return ATTACHMENT_COMPLETE.format(self.cams_url, asset_id, attachment_id)

    def get_search_asset_href(self):
        # return SEARCH_ASSETS.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return SEARCH_ASSETS.format(self._wml_credentials['url'])
        else:
            return SEARCH_ASSETS.format(self.cams_url)

    def get_search_shiny_href(self):
        # return SEARCH_SHINY.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return SEARCH_SHINY.format(self._wml_credentials['url'])
        else:
            return SEARCH_SHINY.format(self.cams_url)

    def get_search_script_href(self):
        # return SEARCH_SCRIPT.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return SEARCH_SCRIPT.format(self._wml_credentials['url'])
        else:
            return SEARCH_SCRIPT.format(self.cams_url)

    def get_model_definition_assets_href(self):
        # return MODEL_DEFINITION_ASSETS.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return MODEL_DEFINITION_ASSETS.format(self._wml_credentials['url'])
        else:
            return MODEL_DEFINITION_ASSETS.format(self.cams_url)

    def get_model_definition_search_asset_href(self):
        # return MODEL_DEFINITION_SEARCH_ASSETS.format(self._wml_credentials['url'])
        if self.cams_url is None:
            return MODEL_DEFINITION_SEARCH_ASSETS.format(self._wml_credentials['url'])
        else:
            return MODEL_DEFINITION_SEARCH_ASSETS.format(self.cams_url)

    def get_wsd_model_href(self):
        return MODEL_DEFINITION_ASSETS.format(self._wml_credentials['url'])

    def get_wsd_model_attachment_href(self):
        return ASSET_FILES.format(self._wml_credentials['url'])

    def get_wsd_asset_search_href(self, asset_type):
        return WSD_SEARCH_ASSETS.format(self._wml_credentials['url'], asset_type)

    def get_wsd_asset_type_href(self):
        return ASSET_TYPE.format(self._wml_credentials['url'])

    def get_wsd_base_href(self):
        return self._wml_credentials['url']

    def get_connections_href(self):
        # return CONNECTION_ASSET.format(self._wml_credentials['url'])
        if self.platform_url is None:
            return CONNECTION_ASSET.format(self._wml_credentials['url'])
        else:
            return CONNECTION_ASSET.format(self.platform_url)

    def get_connection_by_id_href(self, connection_id):
        # return CONNECTION_BY_ID.format(self._wml_credentials['url'], connection_id)
        if self.platform_url is None:
            return CONNECTION_BY_ID.format(self._wml_credentials['url'], connection_id)
        else:
            return CONNECTION_BY_ID.format(self.platform_url, connection_id)

    def get_connection_data_types_href(self):
        # return DATA_SOURCE_TYPE.format(self._wml_credentials['url'])
        if self.platform_url is None:
            return DATA_SOURCE_TYPE.format(self._wml_credentials['url'])
        else:
            return DATA_SOURCE_TYPE.format(self.platform_url)

    def get_sw_spec_href(self, sw_spec_id):
        # return SOFTWARE_SPECIFICATION.format(self._wml_credentials['url'], sw_spec_id)
        if self.platform_url is None:
            return SOFTWARE_SPECIFICATION.format(self._wml_credentials['url'], sw_spec_id)
        else:
            return SOFTWARE_SPECIFICATION.format(self.platform_url, sw_spec_id)

    def get_sw_specs_href(self):
        # return SOFTWARE_SPECIFICATIONS.format(self._wml_credentials['url'])
        if self.platform_url is None:
            return SOFTWARE_SPECIFICATIONS.format(self._wml_credentials['url'])
        else:
            return SOFTWARE_SPECIFICATIONS.format(self.platform_url)

    def get_hw_spec_href(self, hw_spec_id):
        # return HARDWARE_SPECIFICATION.format(self._wml_credentials['url'], hw_spec_id)
        if self.platform_url is None:
            return HARDWARE_SPECIFICATION.format(self._wml_credentials['url'], hw_spec_id)
        else:
            return HARDWARE_SPECIFICATION.format(self.platform_url, hw_spec_id)

    def get_hw_specs_href(self):
        # return HARDWARE_SPECIFICATIONS.format(self._wml_credentials['url'])
        if self.platform_url is None:
            return HARDWARE_SPECIFICATIONS.format(self._wml_credentials['url'])
        else:
            return HARDWARE_SPECIFICATIONS.format(self.platform_url)

    def get_pkg_extn_href(self, pkg_extn_id):
        # return PACKAGE_EXTENSION.format(self._wml_credentials['url'], pkg_extn_id)
        if self.platform_url is None:
            return PACKAGE_EXTENSION.format(self._wml_credentials['url'], pkg_extn_id)
        else:
            return PACKAGE_EXTENSION.format(self.platform_url, pkg_extn_id)

    def get_pkg_extns_href(self):
        # return PACKAGE_EXTENSIONS.format(self._wml_credentials['url'])
        if self.platform_url is None:
            return PACKAGE_EXTENSIONS.format(self._wml_credentials['url'])
        else:
            return PACKAGE_EXTENSIONS.format(self.platform_url)

    def get_project_href(self, project_id):
        # return PROJECT.format(self._wml_credentials['url'], project_id)

        if self.platform_url is None:
            return PROJECT.format(self._wml_credentials['url'], project_id)
        else:
            return PROJECT.format(self.platform_url, project_id)

    def v4ga_cloud_migration_href(self):
        return V4GA_CLOUD_MIGRATION.format(self._wml_credentials['url'])

    def v4ga_cloud_migration_id_href(self, migration_id):
        return V4GA_CLOUD_MIGRATION_ID.format(self._wml_credentials['url'], migration_id)

    def remote_training_system_href(self, remote_training_systems_id):
        if self.platform_url is None:
            return REMOTE_TRAINING_SYSTEM_ID.format(self._wml_credentials['url'], remote_training_systems_id)
        else:
            return REMOTE_TRAINING_SYSTEM_ID.format(self.platform_url, remote_training_systems_id)

    def remote_training_systems_href(self):
        # return PACKAGE_EXTENSIONS.format(self._wml_credentials['url'])
        if self.platform_url is None:
            return REMOTE_TRAINING_SYSTEM.format(self._wml_credentials['url'])
        else:
            return REMOTE_TRAINING_SYSTEM.format(self.platform_url)

    def exports_href(self):
        if self.platform_url is None:
            return EXPORTS.format(self._wml_credentials['url'])
        else:
            return EXPORTS.format(self.platform_url)

    def export_href(self, export_id):
        if self.platform_url is None:
            return EXPORT_ID.format(self._wml_credentials['url'], export_id)
        else:
            return EXPORT_ID.format(self.platform_url, export_id)

    def export_content_href(self, export_id):
        if self.platform_url is None:
            return EXPORT_ID_CONTENT.format(self._wml_credentials['url'], export_id)
        else:
            return EXPORT_ID_CONTENT.format(self.platform_url, export_id)

    def imports_href(self):
        if self.platform_url is None:
            return IMPORTS.format(self._wml_credentials['url'])
        else:
            return IMPORTS.format(self.platform_url)

    def import_href(self, export_id):
        if self.platform_url is None:
            return IMPORT_ID.format(self._wml_credentials['url'], export_id)
        else:
            return IMPORT_ID.format(self.platform_url, export_id)

    def remote_training_systems_href(self):
        return REMOTE_TRAINING_SYSTEM.format(self._wml_credentials['url'] + self.prepend)

    def remote_training_system_href(self, remote_training_systems_id):
        return REMOTE_TRAINING_SYSTEM_ID.format(self._wml_credentials['url'] + self.prepend, remote_training_systems_id)

    def volumes_href(self):
        return VOLUMES.format(self._wml_credentials['url'])

    def volume_href(self,volume_id):
        return VOLUME_ID.format(self._wml_credentials['url'],volume_id)

    def volume_service_href(self,volume_name):
        return VOLUME_SERVICE.format(self._wml_credentials['url'],volume_name)

    def volume_upload_href(self, volume_name):
        return VOLUME_SERVICE_FILE_UPLOAD.format(self._wml_credentials['url'], volume_name)

    # def get_envs_href(self):
    #     return DEPLOYMENT_ENVS_HREF_PATTERN.format(self._wml_credentials['url'])
    #
    # def get_env_href(self, env_id):
    #     return DEPLOYMENT_ENV_HREF_PATTERN.format(self._wml_credentials['url'],env_id)

