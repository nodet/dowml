"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""

from __future__ import print_function


def is_classifier(cls):
    """
    Auxillary function to validate whether a given object is a
    classification-based model. (Can be used for the Fusion Handler as well as
    the Local Training Handler).

    :param cls: Object of interest to identify the type of model.
    :type  cls: `Object`
    :return `bool`
    """
    return cls.loss == 'auto' or cls.loss == 'binary_crossentropy' or \
        cls.loss == 'categorical_crossentropy'
