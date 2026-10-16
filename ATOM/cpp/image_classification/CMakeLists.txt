cmake_minimum_required(VERSION 3.26)

# Define dependencies for external Package
include(FetchContent)
include(cmake/opencv.cmake)
include(cmake/argparse.cmake)

# Define the name of executable
add_executable(image_classification main.cc)

# Update link info for package dependencies: OpenCV
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(image_classification ${OpenCV_LIBS})

# Update link info for dependencies: RBLN
find_package(rbln CONFIG REQUIRED)
target_link_libraries(image_classification rbln::rbln_runtime)

# Update including dependencies: argparse
target_include_directories(image_classification PRIVATE ${argparse_INCLUDE_DIRS})

# Define the name of executable
add_executable(image_classification_async main_async.cc)

# Update link info for package dependencies: OpenCV
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(image_classification_async ${OpenCV_LIBS})

# Update link info for dependencies: RBLN
find_package(rbln CONFIG REQUIRED)
target_link_libraries(image_classification_async rbln::rbln_runtime)

# Update including dependencies: argparse
target_include_directories(image_classification_async PRIVATE ${argparse_INCLUDE_DIRS})
