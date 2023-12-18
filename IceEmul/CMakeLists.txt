ecbuild_add_executable( TARGET daml_ice_emul.x
                        SOURCES iceEmul.cc)
target_compile_features( daml_ice_emul.x  PUBLIC cxx_std_17)
target_link_libraries( daml_ice_emul.x PUBLIC NetCDF::NetCDF_CXX oops atlas "${TORCH_LIBRARIES}")
