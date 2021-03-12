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

from pyspark import SparkContext, SQLContext
from pyspark.sql import DataFrame
from pyspark.sql.types import _parse_datatype_json_string

__all__ = ['DataEdge', "MetaEdge"]

logger = logging.getLogger("mlpipelinepy")


class DataEdge(object):
    """
    DataEdge information
    """

    def __init__(self, s_tuple_id_df):
        try:
            self._java_obj = s_tuple_id_df
            self._id = s_tuple_id_df._1()
            self.j_Data_Frame = s_tuple_id_df._2()
        except Exception as e:
            logger.exception(e)
            raise

    @property
    def id(self):
        return self._id

    @property
    def data_frame(self):
        sc = SparkContext.getOrCreate()
        ctx = SQLContext.getOrCreate(sc)
        return DataFrame(self.j_Data_Frame, ctx)


class MetaEdge(object):
    """
    MetaEdge information
    """

    def __init__(self, s_meta_edge):
        try:
            self._id = str(s_meta_edge._1())
            self._fromNode = None
            self._schema = _parse_datatype_json_string(s_meta_edge._2().json())
        except Exception as e:
            logger.exception(e)
            raise

    @property
    def id(self):
        return self._id

    @property
    def schema(self):
        return self._schema

    def __str__(self):
        return "Id: " + self._id + " Schema: " + str(self._schema)
