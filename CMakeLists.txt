cmake_minimum_required(VERSION 3.8)
project(vision)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp RobotCatcher.cpp RobotCatcher.h test.cpp)
include_directories(/usr/local/include/eigen3)
add_executable(vision ${SOURCE_FILES})
target_link_libraries(vision ${OpenCV_LIBS})