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

    # Test 1: where col3 > set_value
    try:
        df = pd.DataFrame({'col1': [2, 3, 6],
                           'col2': [25, 50, 50],
                           'col3': [51, 60, 101]})
        expected = pd.DataFrame({'col1': [2, 3, 6],
                                 'col2': [25, 50, 50],
                                 'col3': [51, 60, 101],
                                 'prod': [50, 150, 300]})
        result = get_new_df(df, col1='col1', col2='col2', col3='col3', set_value=50)
        pd_testing.assert_frame_equal(result, expected)
        num_passed_tests += 1
    except AssertionError:
        pass

    # Test 2: When col3 <= set_value
    try:
        df = pd.DataFrame({'col1': [1, 2, 3],
                           'col2': [25, 50, 50],
                           'col3': [50, 49, 49]})
        expected = pd.DataFrame({'col1': [1, 2, 3],
                                 'col2': [25, 50, 50],
                                 'col3': [50, 49, 49],
                                 'prod': [50, 49, 49]})
        result = get_new_df(df, col1='col1', col2='col2', col3='col3', set_value=50)
        pd_testing.assert_frame_equal(result, expected)
        num_passed_tests += 1
    except AssertionError:
        pass

    return int(num_passed_tests == total_tests)
