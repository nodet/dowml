"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
JSON based serialization
"""
import jsonpickle
from ibmfl.message.message import Message
from ibmfl.message.serializer import Serializer


class JSONSerializer(Serializer):
    """
    Class for JSON Serializer
    """

    def serialize(self, msg):
        """
        Serialize a message using JSON

        :param msg: message to serialize
        :type msg: `Message`
        :return: serialize byte stream
        :rtype: `b[]`
        """
        msg_header = msg.get_header()
        data = msg.get_data()
        json_str_obj = jsonpickle.encode(
            {
                'header': msg_header,
                'data': data,
            })
        return json_str_obj.encode()  # return a byte stream

    def deserialize(self, serialized_byte_stream):
        """
        Deserialize a byte stream to a message

        :param serialized_byte_stream: byte stream
        :type serialized_byte_stream: `b[]`
        :return: deserialized message
        :rtype: `Message`
        """
        json_str_obj = serialized_byte_stream.decode()
        data_dict = jsonpickle.decode(json_str_obj)

        msg = Message(data=data_dict['data'])
        msg.set_header(data_dict['header'])

        return msg
