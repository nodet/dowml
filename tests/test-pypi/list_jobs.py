import pprint

from dowml.lib import DOWMLLib


def main():
    lib = DOWMLLib()
    pprint.pprint(lib.get_jobs())


if __name__ == '__main__':
    main()
