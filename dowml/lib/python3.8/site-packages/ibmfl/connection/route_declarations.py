"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
Create routes for the server handler
"""
import logging
import json
from ibmfl.message.message_type import MessageType
from ibmfl.message.message import Message

logger = logging.getLogger(__name__)


def default_end_point(message):
    """
    All Requests whose request type is not determined are routed to this method
    Primary function is to log the request received
    :param request: request received by the server
    :type request: `Message`
    :param response: response to send back
    :type response: `Message`
    :param router: router associated with the request
    :type router: `Router`
    :param kwargs: Dictionary of model-specific arguments.
    :type kwargs: `dict`

    """
    # more detailed logging should be done on request message

    logging.info(message.__dict__)

    message.set_data({'status': 'error'})

    return message


def get_aggregator_router(router, agg_proto_handler):
    """
    Route for register party

    :param router: Router object
    :type router: `Router`
    :param agg_proto_handler: ProtoHandler object
    :type agg_proto_handler: `ProtoHandler`
    :return: None
    """
    router.add_routes({
        '{}'.format(MessageType.REGISTER.value):
        agg_proto_handler.register_party,

        '{}'.format(MessageType.TRAIN.value):
        agg_proto_handler.process_model_update_requests,

        'default': default_end_point
    })


def get_party_router(router, party_proto_handler):
    """
    Route for register party

    :param router: Router object
    :type router: `Router`
    :param party_proto_handler: ProtoHandler object
    :type party_proto_handler: `ProtoHandler`
    :return: None
    """
    router.add_routes({
        '{}'.format(MessageType.TRAIN.value): party_proto_handler.handle_async_request,
        '{}'.format(MessageType.SAVE_MODEL.value): party_proto_handler.handle_request,
        '{}'.format(MessageType.EVAL_MODEL.value): party_proto_handler.handle_request,
        '{}'.format(MessageType.SYNC_MODEL.value): party_proto_handler.handle_request,
        '{}'.format(
            MessageType.DECRYPT_FUSED.value): party_proto_handler.handle_request,
        '{}'.format(MessageType.STOP.value): party_proto_handler.handle_request,
        'default': default_end_point
    })
