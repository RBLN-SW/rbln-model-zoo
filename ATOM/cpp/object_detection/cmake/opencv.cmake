include(ExternalProject)
include(ProcessorCount)
include(GNUInstallDirs)

# Setup cmake file
set(OPENCV_SUBBUILD_EP_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv-ep-src)
set(OPENCV_SUBBUILD_EP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv-ep-build)
set(OPENCV_SUBBUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv-build)
set(OPENCV_SUBBUILD_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv)
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/opencv/CMakeLists.txt.in
    ${OPENCV_SUBBUILD_EP_SRC_DIR}/CMakeLists.txt
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
    -G ${CMAKE_GENERATOR}
    -S ${OPENCV_SUBBUILD_EP_SRC_DIR}
    -B ${OPENCV_SUBBUILD_EP_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY
)
ProcessorCount(N)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build . -- -j${N}
    WORKING_DIRECTORY ${OPENCV_SUBBUILD_EP_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY
)

# Add to prefix path
set(OpenCV_DIR ${OPENCV_SUBBUILD_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/opencv4)
