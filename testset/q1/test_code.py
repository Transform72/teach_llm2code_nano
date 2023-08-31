import pandas as pd
import numpy as np
import pandas.testing as pd_testing


def run_tests(generated_code):
    """Assume the generated code is a string"""
    # Make the function available in the current namespace
    local_namespace = {}
    exec(generated_code, globals(), local_namespace)
    get_new_df = local_namespace['get_new_df']  # Retrieve get_new_df from the local namespace

    num_passed_tests = 0  # Number of tests that have passed
    total_tests = 2  # Total number of tests

    # Test 1: Both col2 > set_value1 and col3 < set_value2
    try:
        df = pd.DataFrame({'col1': [25, 30],
                           'col2': [60, 65],
                           'col3': [40, 45]})
        expected = pd.DataFrame({'col1': [35, 40],
                                 'col2': [60, 65],
                                 'col3': [40, 45]})
        result = get_new_df(df, 'col1', 'col2', 'col3', 50, 50)
        pd_testing.assert_frame_equal(result, expected)
        num_passed_tests += 1
    except AssertionError:
        pass

    # Test 2: col2 <= set_value1 or col3 >= set_value2
    try:
        df = pd.DataFrame({'col1': [25, 30],
                           'col2': [40, 30],
                           'col3': [60, 65]})
        expected = pd.DataFrame({'col1': [25, 30],
                                 'col2': [40, 30],
                                 'col3': [60, 65]})
        result = get_new_df(df, 'col1', 'col2', 'col3', 50, 50)
        pd_testing.assert_frame_equal(result, expected)
        num_passed_tests += 1
    except AssertionError:
        pass

    return int(num_passed_tests == total_tests)
