cmake_minimum_required(VERSION 3.26)

# Define dependencies for external Package
include(FetchContent)
include(cmake/opencv.cmake)
include(cmake/argparse.cmake)

# Define the name of executable
add_executable(object_detection main.cc)

# Update link info for package dependencies: OpenCV
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(object_detection ${OpenCV_LIBS})

# Update link info for dependencies: RBLN
find_package(rbln CONFIG REQUIRED)
target_link_libraries(object_detection rbln::rbln_runtime)

# Update including dependencies: argparse
target_include_directories(object_detection PRIVATE ${argparse_INCLUDE_DIRS})

# Define the name of executable
add_executable(object_detection_async main_async.cc)

# Update link info for package dependencies: OpenCV
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(object_detection_async ${OpenCV_LIBS})

# Update link info for dependencies: RBLN
find_package(rbln CONFIG REQUIRED)
target_link_libraries(object_detection_async rbln::rbln_runtime)

# Update including dependencies: argparse
target_include_directories(object_detection_async PRIVATE ${argparse_INCLUDE_DIRS})
