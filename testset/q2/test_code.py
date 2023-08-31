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

    # Test 1: Typical case where col3 > set_value
    try:
        df = pd.DataFrame({'col1': [25, 25, 50],
                           'col2': [30, 35, 55],
                           'col3': [51, 60, 101]})
        expected = pd.DataFrame({'col1': [25, 25, 50],
                                 'col2': [30, 35, 55],
                                 'col3': [51, 60, 101],
                                 'diff': [5, 10, 5]})
        result = get_new_df(df, col1='col1', col2='col2', col3='col3', set_value=50)
        pd_testing.assert_frame_equal(result, expected)
        num_passed_tests += 1
    except AssertionError:
        pass

    # Test 2: When col3 <= set_value
    try:
        df = pd.DataFrame({'col1': [25, 25, 50],
                           'col2': [30, 35, 55],
                           'col3': [49, 50, 50]})
        expected = pd.DataFrame({'col1': [25, 25, 50],
                                 'col2': [30, 35, 55],
                                 'col3': [49, 50, 50],
                                 'diff': [55, 60, 105]})
        result = get_new_df(df, col1='col1', col2='col2', col3='col3', set_value=50)
        pd_testing.assert_frame_equal(result, expected)
        num_passed_tests += 1
    except AssertionError:
        pass

    return int(num_passed_tests == total_tests)
