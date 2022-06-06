 cmake_minimum_required(VERSION 3.22.1)

project(rtsc VERSION 1.6)

list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake)

find_package(Trimesh2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Glui REQUIRED)

add_executable(rtsc isolines.cc rtsc.cc apparentridge.cc)
target_include_directories(rtsc PUBLIC . ${TRIMESH_DIR}/include ${CMAKE_CURRENT_LIST_DIR})
target_link_directories(rtsc PUBLIC ${TRIMESH_DIR}/build/libsrc)
target_link_libraries(rtsc PUBLIC ${GLUT_LIBRARIES} Glui::glui ${OPENGL_LIBRARIES} libtrimesh2)