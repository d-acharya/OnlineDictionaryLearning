cmake_minimum_required(VERSION 2.8)
project(ODLForSparseCoding)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-O3 -mavx2 -mfma -m64 -std=c++0x -ffast-math")

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/lars_light/src)
include_directories(${CMAKE_SOURCE_DIR}/src/lars_light/lib)

add_library(mathOperations src/lars_light/src/mathOperations.cpp)
add_library(lars src/lars_light/src/lars.cpp)
add_library(OnlineDictionaryLearning src/OnlineDictionaryLearning.cpp)

add_executable(test_odl test_odl.cpp)
target_link_libraries(test_odl OnlineDictionaryLearning)
target_link_libraries(test_odl lars)
target_link_libraries(test_odl mathOperations)

add_executable(denoising denoising.cpp)

target_link_libraries(denoising ${OpenCV_LIBS})
target_link_libraries(denoising OnlineDictionaryLearning)
target_link_libraries(denoising mathOperations)
target_link_libraries(denoising lars)
