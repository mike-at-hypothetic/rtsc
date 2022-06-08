add_library(librtsc "")
target_sources(librtsc PRIVATE isolines.cc)
target_sources(librtsc PUBLIC rtsc.h)
target_link_libraries(librtsc PUBLIC Trimesh2::trimesh2)