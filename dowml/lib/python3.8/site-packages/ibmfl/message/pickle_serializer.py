"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
Pickle based serialization
"""
import pickle
from ibmfl.message.message import Message
from ibmfl.message.serializer import Serializer


class PickleSerializer(Serializer):
    """
    Class for Pickle based serialization
    """

    def serialize(self, msg):
        """
        Serialize a message using pickle

        :param msg: message to serialize
        :type msg: `Message`
        :return: serialize byte stream
        :rtype: `b[]`
        """
        msg_header = msg.get_header()
        serialized_data = msg.get_data()  # need to serialize the data
        return pickle.dumps({'header': msg_header,
                             'data': serialized_data,
                             })

    def deserialize(self, serialized_byte_stream):
        """
        Deserialize a byte stream to a message

        :param serialized_byte_stream: byte stream
        :type serialized_byte_stream: `b[]`
        :return: deserialized message
        :rtype: `Message`
        """
        data_dict = pickle.loads(serialized_byte_stream)
        if 'MSG_LEN|' in data_dict:
            msg_length = int(data_dict.split('|')[1])
            return msg_length

        msg = Message(data=data_dict['data'])
        msg.set_header(data_dict['header'])

        return msg
