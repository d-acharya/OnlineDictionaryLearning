cmake_minimum_required(VERSION 2.8)
project(ODLForSparseCoding)

find_package(OpenCV REQUIRED)
#find_package(MKL Required)

set(MKLROOT /opt/intel/mkl/)
set(CMAKE_CXX_FLAGS "-DMKL_ILP64 -m64 -I${MKLROOT}/include")

#if(NOT MKL_FOUND)
	#message(FATAL_ERROR "MKL Not Found")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
#endif()

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-g -O3 -mavx -mfma -m64 -std=c++0x")
endif()



include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/lars_light/src)
include_directories(${CMAKE_SOURCE_DIR}/src/lars_light/lib)
include_directories(${MKLROOT}/include)

add_library(mathOperations src/lars_light/src/mathOperations.cpp)
add_library(lars src/lars_light/src/lars.cpp)
add_library(OnlineDictionaryLearning src/OnlineDictionaryLearning.cpp)

add_executable(test_odl test_odl.cpp)
target_link_libraries(test_odl OnlineDictionaryLearning)
target_link_libraries(test_odl lars)
target_link_libraries(test_odl mathOperations)
target_link_libraries(test_odl -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl)

if(OpenCV_FOUND)
	add_executable(denoising denoising.cpp)
	target_link_libraries(denoising ${OpenCV_LIBS})
	target_link_libraries(denoising OnlineDictionaryLearning)
	target_link_libraries(denoising mathOperations)
	target_link_libraries(denoising lars)	
else()
	message(STATUS "Compiled without OpenCV")
endif()
