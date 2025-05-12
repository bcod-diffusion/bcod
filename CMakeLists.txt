cmake_minimum_required(VERSION 3.18)
project(bcod VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(Boost 1.74.0 REQUIRED COMPONENTS system filesystem thread)
find_package(Eigen3 3.3.7 REQUIRED)
find_package(OpenCV 4.5.0 REQUIRED)
find_package(CUDA 11.0 REQUIRED)
find_package(TensorRT 8.0 REQUIRED)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Add library target
add_library(bcod SHARED
    src/belief_rasteriser.cpp
    src/student_planner.cpp
    src/sac_scheduler.cpp
    src/utils.cpp
)

# Link libraries
target_link_libraries(bcod
    PRIVATE
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# Set compile definitions
target_compile_definitions(bcod
    PRIVATE
    BOOST_ALL_DYN_LINK
    EIGEN_MPL2_ONLY
    TORCH_CUDA_VERSION=110
    TENSORRT_VERSION=8
    OPENCV_VERSION=4
)

# Set compile options
target_compile_options(bcod
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O3
    -march=native
    -ffast-math
    -fopenmp
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        -Xcompiler=-O3
        -Xcompiler=-march=native
        -Xcompiler=-ffast-math
        -Xcompiler=-fopenmp
    >
)

# Set CUDA specific options
set_target_properties(bcod PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86"
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)

# Install rules
install(TARGETS bcod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/bcod
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Add tests
enable_testing()
add_subdirectory(tests)

# Add examples
add_subdirectory(examples)

# Add documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Add version info
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bcod/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/bcod/version.hpp
    @ONLY
) 