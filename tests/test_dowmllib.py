from unittest import TestCase

from dowml.dowmllib import DOWMLLib


class TestImport(TestCase):

    def test_dowmllib_was_imported(self):
        # The point is only to confirm that 'dowmllib' still
        # exists as a submodule in dowml, for compatibility
        self.assertEqual('DOWMLLib', DOWMLLib.__name__)
