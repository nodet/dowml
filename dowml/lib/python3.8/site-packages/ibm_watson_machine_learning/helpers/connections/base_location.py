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

from abc import ABC

__all__ = [
    "BaseLocation"
]


class BaseLocation(ABC):
    """
    Base class for storage Location.
    """

    def to_dict(self) -> dict:
        """Return a json dictionary representing this model."""
        return vars(self)

    def _get_file_size(self, **kwargs) -> 'int':
        raise NotImplementedError
