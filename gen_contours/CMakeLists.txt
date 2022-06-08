add_executable(gen_contours "")
target_sources(gen_contours PRIVATE gen_contours.cpp)
target_include_directories(gen_contours PUBLIC . ${CMAKE_CURRENT_LIST_DIR} ../librtsc)
target_link_libraries(gen_contours PUBLIC librtsc Trimesh2::trimesh2)