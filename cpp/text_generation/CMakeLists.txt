cmake_minimum_required(VERSION 3.26)

# Define dependencies for external Package
include(FetchContent)
include(cmake/argparse.cmake)

# Collect all source files
file(GLOB SOURCE_FILES "*.cc")

# Define executable
add_executable(text_generation llama_main.cc ${SOURCE_FILES})
# Backward-compatible target name
add_executable(llama_binding ALIAS text_generation)

# Link RBLN runtime
find_package(rbln CONFIG REQUIRED)
target_link_libraries(text_generation rbln::rbln_runtime)

# Update including dependencies: argparse
target_include_directories(text_generation PRIVATE ${argparse_INCLUDE_DIRS})

# Add header files directory
target_include_directories(text_generation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
