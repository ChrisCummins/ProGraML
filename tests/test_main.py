import os
import sys

import pytest


def main():
    """The main entry point for the pytest runner.

    An example file which uses this:

        from compiler_gym.frontend.gym_interface.util.test_main import main

        def test_foo():
            assert 1 + 1 == 2

        if __name__ == "__main__":
            main()

    In the above, the single test_foo test will be executed.
    """
    pytest_args = sys.argv + ["-vv"]
    # Support for sharding. If a py_test target has the shard_count attribute
    # set (in the range [1,50]), then the pytest-shard module is used to divide
    # the tests among the shards. See https://pypi.org/project/pytest-shard/
    sharded_test = os.environ.get("TEST_TOTAL_SHARDS")
    if sharded_test:
        num_shards = int(os.environ["TEST_TOTAL_SHARDS"])
        shard_index = int(os.environ["TEST_SHARD_INDEX"])
        pytest_args += [f"--shard-id={shard_index}", f"--num-shards={num_shards}"]
    else:
        pytest_args += ["-p", "no:pytest-shard"]

    sys.exit(pytest.main(pytest_args))
