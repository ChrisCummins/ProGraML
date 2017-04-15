# sources and data glob.
sh_library(
    name = 'cldrive',
    srcs = glob([
        'bin/cldrive',
        'cldrive/*.py',
        'Makefile',
        'pytest.ini',
        'requirements.txt',
        'setup.cfg',
        'setup.py',
        'tests/*.py',
        'tests/data/**/*',
    ]),
    visibility = ['//visibility:public'],
)

# a script which sets up a virtualenv and runs the test suite.
sh_test(
    name = 'main',
    srcs = ['tests/.runner.sh'],
    args = ['src/cldrive', 'python3.6'],
    deps = [':cldrive'],
    timeout = 'eternal',
)
