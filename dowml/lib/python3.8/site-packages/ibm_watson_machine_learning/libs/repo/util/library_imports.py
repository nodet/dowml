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

import pkg_resources
from ibm_watson_machine_learning.libs.repo.util.base_singleton import BaseSingleton
from ibm_watson_machine_learning.libs.repo.base_constants import *
import subprocess
import json
class LibraryChecker(BaseSingleton):
    def __init__(self):
        self.supported_libs = [PYSPARK,
                               SCIKIT,
                               PANDAS,
                               XGBOOST,
                               MLPIPELINE,
                               IBMSPARKPIPELINE,
                               TENSORFLOW]
        self.installed_libs = {PYSPARK: False,
                               SCIKIT: False,
                               PANDAS: False,
                               XGBOOST: False,
                               MLPIPELINE: False,
                               TENSORFLOW: False,
                               IBMSPARKPIPELINE: False}
        atleast_one_lib_installed = self._check_if_lib_installed(self.supported_libs)

        if not atleast_one_lib_installed:
            supported_lib_str = self.supported_libs[0]
            lib_num = len(self.supported_libs)
            for i in range(1, lib_num-1):
                supported_lib_str += ', ' + self.supported_libs[i]
            supported_lib_str += ' and ' + self.supported_libs[lib_num-1]
            raise ImportError("The system lacks installations of " + supported_lib_str +
                              ". At least one of the libraries is required for the repository-client to be used")

    def _check_if_lib_installed(self, lib_names):
        import sys
        atleast_one_lib_installed = False

        # Using try except method as these libraries aren't present in the
        # PyPI repository for pip
        try:
            import pyspark
            self.installed_libs[PYSPARK] = True
            atleast_one_lib_installed = True
        except ImportError:
            pass
        try:
            import mlpipelinepy
            self.installed_libs[MLPIPELINE] = True
            atleast_one_lib_installed = True
        except ImportError:
            pass
        try:
            import pipeline
            self.installed_libs[IBMSPARKPIPELINE] = True
            atleast_one_lib_installed = True
        except ImportError:
            pass

        installed_pkgs = pkg_resources.working_set
        pkg_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_pkgs])
        for name in lib_names:
            for pkg in pkg_list:
                if name in str(pkg):
                    self.installed_libs[name] = True
                    atleast_one_lib_installed = True
                if 'scikit-learn' == str(pkg):
                    import sklearn
                    import sys
                    if sklearn.__version__ == '0.23.0' and sys.version_info <= (3, 7):
                        raise Exception(' Scikit learn version 0.23.0 is not supported, Please downgrade scikit version to a lower version and re-try. ')

            # If above asked package is not found using pip, check if it is installed using 'conda'
            # To check it, first verify if conda is installed.
            if not (sys.platform.startswith("win") or sys.platform.startswith("cygwin")):
                if (not self.installed_libs[name]):
                    chk_conda = subprocess.Popen(["which", "conda"], stdout=subprocess.PIPE)
                    stdout, stderr = chk_conda.communicate()
                    if (stdout.decode() != ''):  # if string not '', means conda is installed.
                        try:
                            p = subprocess.Popen(["conda", "list", name, "--json", ], stdout=subprocess.PIPE)
                            stdout, stderr = p.communicate()
                            val = json.loads(stdout.decode())
                            if len(val) > 0:
                                self.installed_libs[name] = True
                                atleast_one_lib_installed = True
                        except OSError:
                            pass
        return atleast_one_lib_installed

    def check_lib(self, lib_name):
        lib_display_names={PYSPARK: DISPLAY_PYSPARK,
                           SCIKIT: DISPLAY_SCIKIT,
                           PANDAS: DISPLAY_PANDAS,
                           XGBOOST: DISPLAY_XGBOOST,
                           MLPIPELINE: DISPLAY_MLPIPELINE,
                           TENSORFLOW: DISPLAY_TENSORFLOW,
                           IBMSPARKPIPELINE: DISPLAY_IBMSPARKPIPELINE}
        if not self.installed_libs[lib_name]:
            raise NameError('{} Library is not installed. Please install it and execute the command'.
                            format(lib_display_names[lib_name]))