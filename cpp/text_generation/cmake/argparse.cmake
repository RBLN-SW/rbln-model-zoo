FetchContent_Declare(argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.0
)
FetchContent_MakeAvailable(argparse)
set(argparse_INCLUDE_DIRS ${argparse_SOURCE_DIR}/include)
