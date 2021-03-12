import unittest
from lale.operators import BasePipeline
test_case = unittest.TestCase()


def is_lale_pipeline_type(pipeline: object) -> None:
    print("Testing if object is of lale pipeline type")
    test_case.assertIsInstance(pipeline, BasePipeline)
