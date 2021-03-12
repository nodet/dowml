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

from .swagger_client.api_client import ApiClient
from .swagger_client.rest import ApiException
import os
import ssl

import certifi

try:
    import urllib3
except ImportError:
    raise ImportError('urllib3 is missing')

try:
    # for python3
    from urllib.parse import urlencode
except ImportError:
    # for python2
    from urllib import urlencode
http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)


class MLApiClient(ApiClient):
    """

    Class extending ApiClient.

    """
    def __init__(self, repository_path):
        super(MLApiClient, self).__init__(repository_path)
        self.repository_path = repository_path

    def download_file(self, path, presigned_url, query_params, header_params):

        if 'DEPLOYMENT_PLATFORM' in os.environ and os.environ['DEPLOYMENT_PLATFORM'] == 'private':
            http = urllib3.PoolManager(
                cert_reqs=ssl.CERT_NONE,
                ca_certs=certifi.where()
            )
        else:
            http = urllib3.PoolManager(
                cert_reqs='CERT_REQUIRED',
                ca_certs=certifi.where()
            )

        tmp_headers = self.default_headers.copy()
        tmp_headers.update(header_params)
        if presigned_url == 'true':
            path = path.replace('%2F', '/')
            tmp_headers = header_params.copy()

        r = http.request(
            'GET',
            '{}'.format(self.host+path, urlencode(query_params)),
            headers=tmp_headers,
            preload_content=False
            )
        if r.status == 200:
            return r
        else:
            raise ApiException(r.status, 'No content for: {}'.format(path))

    def download_file_v4(self, path, presigned_url, query_params, header_params):
        if 'DEPLOYMENT_PLATFORM' in os.environ and os.environ['DEPLOYMENT_PLATFORM'] == 'private':
            http = urllib3.PoolManager(
                cert_reqs=ssl.CERT_NONE,
                ca_certs=certifi.where()
            )
        else:
            http = urllib3.PoolManager(
                cert_reqs='CERT_REQUIRED',
                ca_certs=certifi.where()
            )
        tmp_headers = self.default_headers.copy()
        tmp_headers.update(header_params)
        if presigned_url == 'true':
            path = path.replace('%2F', '/')
        tmp_headers = header_params.copy()
        tmp_headers.update({'Content-Type': 'application/json'})
        tmp_headers.update({'ML-Instance-ID': 'invalid', 'x-wml-internal-switch-to-new-v4': 'true'})

        r = http.request(
            'GET',
            '{}'.format(self.host + path),
            headers=tmp_headers,
            preload_content=False,
            fields=query_params
        )
        if r.status == 200:
            return r
        else:
            raise ApiException(r.status, 'No content for: {}'.format(path))
