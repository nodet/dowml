"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
import os
import pathlib
from environs import Env

env = Env()
env.read_env()

# export FL_WORKING_DIR=/home/username/
# export FL_MODEL_DIR=/home/username/model


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


with env.prefixed("FL_"):
    working_directory = env("WORKING_DIR", os.getcwd())
    create_dir(working_directory)

    model_directory = env("MODEL_DIR", working_directory)
    create_dir(model_directory)
