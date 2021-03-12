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

from typing import Union
import sys

from ibm_watson_machine_learning.utils.autoai.utils import try_import_tqdm

try_import_tqdm()

from tqdm import tqdm as TQDM

__all__ = [
    "ProgressBar"
]


class ProgressBar(TQDM):
    """
    Progress Bar class for handling progress bar display. It is based on 'tqdm' class, could be extended.

    Parameters
    ----------
    desc: str, optional
        Description string to be added as a prefix to progress bar.

    total: int, optional
        Total length of the progress bar.
    """

    def __init__(self, ncols: Union[str, int], position: int = 0, desc: str = None, total: int = 100,
                 leave: bool = True, bar_format: str = '{desc}: {percentage:3.0f}%|{bar}|') -> None:
        # note: to see possible progress bar formats please look at super 'bar_format' description
        super().__init__(desc=desc, total=total, leave=leave, position=position, ncols=ncols, file=sys.stdout,
                         bar_format=bar_format)
        self.total = total
        self.previous_message = None
        self.counter = 0
        self.progress = 0

    def increment_counter(self, progress: int = 5) -> None:
        """
        Increments internal counter and waits for specified time.

        Parameters
        ----------
        progress: int, optional
            How many steps at a time progress bar will increment.
        """
        self.progress = progress
        self.counter += progress

    def reset_counter(self) -> None:
        """
        Restart internal counter
        """
        self.counter = 0

    def update(self):
        """
        Updates the counter with specific progress.
        """
        super().update(n=self.progress)

    def last_update(self):
        """Fill up the progress bar till the end, this was the last run."""
        super().update(n=self.total - self.counter)
