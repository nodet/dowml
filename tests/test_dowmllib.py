from unittest import TestCase

from dowml.dowmllib import DOWMLLib


class TestImport(TestCase):

    def test_can_instantiate_lib(self):
        # The point is only to confirm that 'dowmllib' still
        # exists as a submodule in dowml, for compatibility
        _ = DOWMLLib()
