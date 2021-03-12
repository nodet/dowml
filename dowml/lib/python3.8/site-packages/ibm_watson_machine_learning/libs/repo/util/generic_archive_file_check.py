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

class GenericArchiveFrameworkCheck(object):
    @staticmethod
    def is_archive_framework(name):
        if (re.match('spss-modeler', name) is not None) or (re.match('pmml', name) is not None) \
          or (re.match('caffe', name) is not None) or (re.match('caffe2', name) is not None) \
          or (re.match('torch',name) is not None) \
          or (re.match('pytorch', name) is not None) or (re.match('blueconnect', name) is not None) \
          or (re.match('mxnet', name) is not None) or (re.match('theano', name) is not None) \
          or (re.match('darknet', name) is not None):
          return True
        else:
         return False
