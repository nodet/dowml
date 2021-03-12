"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
 An enumeration class for the message type field which describe what
 kind of data is being sent inside the Message
"""
from enum import Enum

__author__ = "Supriyo Chakraborty, Shalisha Witherspoon, Dean Steuer"


class MessageType(Enum):
    """
    Message types for communication between party and aggregator
    """
    MODEL_UPDATE = 1
    MODEL_HYPERPARAMETERS = 2
    MODEL_PARAMETERS = 3
    REQUEST_MODEL_HYPERPARAMETERS = 4
    REQUEST_MODEL_UPDATE = 5
    REGISTER = 6
    TRAIN = 7
    SAVE_MODEL = 8
    PREDICT_MODEL = 9
    EVAL_MODEL = 10
    ACK = 11
    SYNC_MODEL = 12
    DECRYPT_FUSED = 13
    STOP = 14
    ERROR_AUTH = 400
