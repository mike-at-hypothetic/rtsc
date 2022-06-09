add_library(librtsc "")
target_sources(librtsc PRIVATE apparentridge.cc isolines.cc)
target_sources(librtsc PUBLIC apparentridge.h rtsc.h)
target_link_libraries(librtsc PUBLIC Trimesh2::trimesh2)