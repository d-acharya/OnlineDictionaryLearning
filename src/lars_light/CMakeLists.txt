cmake_minimum_required (VERSION 2.6)
project (test_lars)

add_library(mathOperations src/mathOperations.cpp)

set(test_source_files test.cpp)
add_executable(test_lars ${test_source_files})

target_link_libraries(test_lars mathOperations)
