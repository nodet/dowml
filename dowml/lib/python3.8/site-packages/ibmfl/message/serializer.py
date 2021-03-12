"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
from abc import ABC, abstractmethod


class Serializer(ABC):
    """
    Abstract class for Serializer
    """
    @abstractmethod
    def serialize(self, msg):
        """
        Serialize a message

        :param msg: message to serialize
        :type msg: `Message`
        :return: serialized byte stream
        :rtype: `b[]`
        """
        pass

    @abstractmethod
    def deserialize(self, serialized_byte_stream):
        """
        Deserialize a byte stream to a message

        :param serialized_byte_stream: byte stream
        :type serialized_byte_stream: `b[]`
        :return: deserialized message
        :rtype: `Message`
        """
        pass
