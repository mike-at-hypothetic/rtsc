add_executable(rtsc_gui rtsc.cc apparentridge.cc)

target_include_directories(rtsc_gui PUBLIC . ${CMAKE_CURRENT_LIST_DIR} ../librtsc)
target_link_libraries(rtsc_gui PUBLIC librtsc ${GLUT_LIBRARIES} Glui::glui ${OPENGL_LIBRARIES} Trimesh2::trimesh2)