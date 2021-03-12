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

import logging
import sys

__all__ = [
    "WMLClientError",
    "MissingValue",
    "MissingMetaProp",
    "NotUrlNorUID",
    "NoWMLCredentialsProvided",
    "ApiRequestFailure",
    "UnexpectedType",
    "ForbiddenActionForPlan",
    "NoVirtualDeploymentSupportedForICP",
    "MissingArgument",
    "WrongEnvironmentVersion"
]


class WMLClientError(Exception):
    def __init__(self, error_msg, reason = None):
        self.error_msg = error_msg
        self.reason = reason
        logging.getLogger(__name__).warning(self.__str__())
        logging.getLogger(__name__).debug(str(self.error_msg) + ('\nReason: ' + str(self.reason) if sys.exc_info()[0] is not None else ''))

    def __str__(self):
        return str(self.error_msg) + ('\nReason: ' + str(self.reason) if self.reason is not None else '')


class MissingValue(WMLClientError, ValueError):
    def __init__(self, value_name, reason = None):
        WMLClientError.__init__(self, 'No \"' + value_name + '\" provided.', reason)


class MissingMetaProp(MissingValue):
    def __init__(self, name, reason = None):
        WMLClientError.__init__(self, 'Missing meta_prop with name: \'{}\'.'.format(name), reason)


class NotUrlNorUID(WMLClientError, ValueError):
    def __init__(self, value_name, value, reason = None):
        WMLClientError.__init__(self, 'Invalid value of \'{}\' - it is not url nor uid: \'{}\''.format(value_name, value), reason)


class NoWMLCredentialsProvided(MissingValue):
    def __init__(self, reason = None):
        MissingValue.__init__(self, 'WML credentials', reason)


class ApiRequestFailure(WMLClientError):
    def __init__(self, error_msg, response, reason = None):
        WMLClientError.__init__(self, '{} ({} {})\nStatus code: {}, body: {}'.format(error_msg, response.request.method, response.request.url, response.status_code, response.text if response.apparent_encoding is not None else '[binary content, ' + str(len(response.content)) + ' bytes]'), reason)


class UnexpectedType(WMLClientError, ValueError):
    def __init__(self, el_name, expected_type, actual_type):
        WMLClientError.__init__(self, 'Unexpected type of \'{}\', expected: {}, actual: \'{}\'.'.format(el_name, '\'{}\''.format(expected_type) if type(expected_type) == type else expected_type, actual_type))


class ForbiddenActionForPlan(WMLClientError):
    def __init__(self, operation_name, expected_plans, actual_plan):
        WMLClientError.__init__(self, 'Operation \'{}\' is available only for {} plan, while this instance has \'{}\' plan.'.format(operation_name, ('one of {} as'.format(expected_plans) if len(expected_plans) > 1 else '\'{}\''.format(expected_plans[0])) if type(expected_plans) is list else '\'{}\''.format(expected_plans), actual_plan))


class NoVirtualDeploymentSupportedForICP(MissingValue):
    def __init__(self, reason = None):
        MissingValue.__init__(self, 'No Virtual deployment supported for ICP', reason)


class MissingArgument(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Argument: {value_name} missing.", reason)


class WrongEnvironmentVersion(WMLClientError, ValueError):
    def __init__(self, used_version, environment_name, supported_versions):
        WMLClientError.__init__(self, "Version used in credentials not supported in this environment",
                                reason=f"Version {used_version} isn't supported in "
                                       f"{environment_name} environment, "
                                       f"select from {supported_versions}")