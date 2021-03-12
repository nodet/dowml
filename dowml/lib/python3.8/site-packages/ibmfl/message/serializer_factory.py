"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
Serialization factory provides a way to create a serializer
and deserializer to convert byte streams to a message and
vice versa
"""
from ibmfl.message.json_serializer import JSONSerializer
from ibmfl.message.pickle_serializer import PickleSerializer
from ibmfl.message.serializer_types import SerializerTypes


class SerializerFactory(object):
    """
    Class for a factory to serialize and deserialize
    """
    def __init__(self, serializer_type):
        """
        Creates an object of `SerializerFactory` class

        :param serializer_type: type of seriaze and deserialize
        :type serializer_type: `Enum`
        """
        self.serializer = None
        if serializer_type is SerializerTypes.PICKLE:
            self.serializer = PickleSerializer()
        elif serializer_type is SerializerTypes.JSON_PICKLE:
            self.serializer = JSONSerializer()

    def build(self):
        """
        Returns a serializer

        :param: None
        :return: serializer
        :rtype: `Serializer`
        """
        return self.serializer
