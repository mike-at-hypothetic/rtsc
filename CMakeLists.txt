 cmake_minimum_required(VERSION 3.22.1)

 set(CMAKE_CXX_STANDARD 17)
 
 project(rtsc VERSION 1.6)

list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake)

find_package(Trimesh2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Glui REQUIRED)

add_subdirectory(librtsc)
add_subdirectory(rtsc_gui)
add_subdirectory(gen_contours)
