# clgen sources and data glob.
sh_library(
    name = 'clgen',
    srcs = glob([
        'bin/*',
        'clgen/**/*.py',
        'clgen/data/**/*',
        'configure',
        'make/**/*',
        'Makefile',
        'native/*.cpp',
        'setup.py',
        'tests/**/*',
    ]),
    visibility = ['//visibility:public'],
)

# a script which sets up a virtualenv and runs the test suite.
sh_test(
    name = 'main',
    srcs = ['tests/.runner.sh'],
    args = ['src/clgen', 'python3.6'],
    deps = [':clgen'],
    timeout = 'eternal',
)
