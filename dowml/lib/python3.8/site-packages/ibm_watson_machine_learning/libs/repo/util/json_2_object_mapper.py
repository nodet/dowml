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

import json
from collections import namedtuple


class Json2ObjectMapper(object):
    @staticmethod
    def read(json_holder):  # TODO map
        if json_holder is None:
            return {}
        elif isinstance(json_holder, str):
            return json.loads(json_holder)
        else:
            return json_holder  # TODO json.loads(json_str, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    @staticmethod
    def to_dict(json_str):
        if isinstance(json_str, str):
            return json.loads(json_str)
        else:
            raise ValueError('Incorrect type')

    @staticmethod
    def to_object(json_holder):
        if isinstance(json_holder, str):
            return json.loads(json_holder, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        elif isinstance(json_holder, object):
            return json_holder
        else:
            raise ValueError('Incorrect type')
