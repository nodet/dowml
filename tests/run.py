import logging
import pprint
import types

import requests

from dowmllib import DOWMLLib

orig_requests_session_send = requests.Session.send


def mocked_requests_session_send(*arguments, **kwargs):
    session, prepared_request = arguments
    method = prepared_request.method
    url = prepared_request.url
    #       2021-02-17 16:59:39,710
    print(f'           {method} {url}')
    resp = orig_requests_session_send(*arguments, **kwargs)
    # print(f'                        {resp.status_code}')
    return resp


old_params = None


def patch_params():
    global old_params
    result = old_params()
    # Beware: the parameter list must not have spaces!
    result['include'] = 'solve_parameters,solve_state,status'
    return result


def main():
    global old_params
    logging.basicConfig(force=True, format='%(asctime)s %(message)s',
                        level=logging.DEBUG)
    requests.Session.send = mocked_requests_session_send
    lib = DOWMLLib()
    lib._get_or_make_client()

    old_params = lib._client._params
    try:
        lib._client._params = patch_params
        details = lib.get_job_details('4f90302a-930d-4b97-b98a-d5db8c83b004')
    finally:
        lib._client._params = old_params

    pprint.pprint(details)

#
# class Lib:
#     def foo(self):
#         return 'foo'
#
#     def print(self):
#         f = self.foo()
#         print(f)
#
#
# old_foo = None
# bar = 'bar'
#
#
# def new_foo(self):
#     global old_foo
#     global bar
#     f = old_foo()
#     return f + bar
#
#
# def main():
#     global old_foo
#     func_type = types.MethodType
#
#     my_lib = Lib()
#     old_foo = my_lib.foo
#     my_lib.foo = func_type(new_foo, my_lib)
#     my_lib.print()


if __name__ == '__main__':
    main()
