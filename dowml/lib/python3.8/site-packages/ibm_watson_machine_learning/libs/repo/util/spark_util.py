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

from .library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[PYSPARK]:
    from pyspark.ml import Pipeline, PipelineModel


class SparkUtil(object):
    DEFAULT_LABEL_COL = 'label'

    @staticmethod
    def get_label_col(spark_artifact):
        lib_checker.check_lib(PYSPARK)
        if isinstance(spark_artifact, PipelineModel):
            pipeline = Pipeline(stages=spark_artifact.stages)
            return SparkUtil.get_label_col_from__stages(pipeline.getStages())
        elif isinstance(spark_artifact, Pipeline):
            return SparkUtil.get_label_col_from__stages(spark_artifact.getStages())
        else:
            return SparkUtil.DEFAULT_LABEL_COL

    @staticmethod
    def get_label_col_from__stages(stages):
        lib_checker.check_lib(PYSPARK)
        label = SparkUtil._get_label_col_from_python_stages(stages)

        if label == SparkUtil.DEFAULT_LABEL_COL:
            label = SparkUtil._get_label_col_from_java_stages(stages)

        return label

    @staticmethod
    def _get_label_col_from_python_stages(stages):
        try:
            label_col = stages[-1].getLabelCol()
        except Exception as ex:
            label_col = SparkUtil.DEFAULT_LABEL_COL

        reversed_stages = stages[:]
        reversed_stages.reverse()

        for stage in reversed_stages[1:]:
            try:
                if stage.getOutputCol() == label_col:
                    label_col = stage.getInputCol()
            except Exception as ex:
                pass

        return label_col

    @staticmethod
    def _get_label_col_from_java_stages(stages):
        try:
            label_col = stages[-1]._call_java("getLabelCol")
        except Exception as ex:
            label_col = SparkUtil.DEFAULT_LABEL_COL

        reversed_stages = stages[:]
        reversed_stages.reverse()

        for stage in reversed_stages[1:]:
            try:
                if stage._call_java("getOutputCol") == label_col:
                    label_col = stage._call_java("getInputCol")
            except Exception as ex:
                pass

        return label_col
