project(BasicObjectRecognition)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_BUILD_TYPE "Release")

set(STD_FLAGS_DEBUG "-Wall -g")
set(STD_FLAGS_RELEASE "-Wall -export-dynamic -march=nocona -mfpmath=sse -ftree-vectorize -fextended-identifiers -O3") 

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(best_face_pose src/main.cpp)
target_link_libraries(best_face_pose ${OpenCV_LIBRARIES})
