# Find required packages
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

# Add test executable
add_executable(bcod_tests
    belief_rasteriser_test.cpp
    student_planner_test.cpp
    sac_scheduler_test.cpp
)

# Link against required libraries
target_link_libraries(bcod_tests
    PRIVATE
    bcod
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
)

# Include directories
target_include_directories(bcod_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Add test
add_test(NAME bcod_tests COMMAND bcod_tests) 