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


class MetaProps:
    """
    Holder for props used during creation of ML artifacts.

    :param map meta: Map of pair key and value where key is taken from MetaNames.
    """
    def __init__(self, meta):
        self.meta = meta

    def available_props(self):
        """Return list of strings with names of available props."""
        return self.meta.keys()

    def prop(self, name):
        """Get prop value by name."""
        return self.meta.get(name)

    def merge(self, other):
        """Merge other MetaProp object to first one. Modify first MetaProp object, doesn't return anything."""
        self.meta.update(other.meta)

    def add(self, name, value):
        """
        Add new prop.

        :param str name: Key for value. Should be one of the values from MetaNames.
        :param object value: Any type of object
        """
        self.meta[name] = value

    def get(self):
        """returns meta prop dict"""
        return self.meta
