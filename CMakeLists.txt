cmake_minimum_required(VERSION 2.8)
project(ODLForSparseCoding)

find_package(OpenCV REQUIRED)

include_directories(src)
include_directories(src/lars_ligh/src)

add_library(utilities src/utilities.cpp)
add_library(OnlineDictionaryLearning src/OnlineDictionaryLearning.cpp)
add_library(mathOperations src/lars_light/src/mathOperations.cpp)

add_executable(denoising denoising.cpp)

target_link_libraries(denoising ${OpenCV_LIBS})
target_link_libraries(denoising utilities)
target_link_libraries(denoising mathOperations)
target_link_libraries(denoising lars)

