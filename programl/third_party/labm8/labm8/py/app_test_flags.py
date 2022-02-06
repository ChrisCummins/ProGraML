"""Flags for :app_test.py.

A quirk in the combination of pytest and absl flags is that you can't define
a flag in the same file that you invoke pytest.main(). This is because the
pytest collector re-imports the file, causing absl to error because the flags
have already been defined.
"""
from labm8.py import app, sqlutil

FLAGS = app.FLAGS

app.DEFINE_string("string_flag", None, "A string argument")

app.DEFINE_output_path(
    "output_path_flag",
    "/tmp/temporary_file",
    "A path argument",
)

app.DEFINE_integer(
    "int_flag_with_validator",
    1,
    "An int flag",
    validator=lambda val: 0 < val < 10,
)


class MockDatabase(sqlutil.Database):
    """Database for testing."""

    def __init__(self, url, *args, **kwargs):
        super(MockDatabase, self).__init__(url, sqlutil.Base(), *args, **kwargs)


app.DEFINE_database(
    "database_flag",
    MockDatabase,
    "sqlite://",
    "A database argument.",
)
