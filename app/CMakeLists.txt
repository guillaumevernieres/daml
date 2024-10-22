# Seaice emulator
ecbuild_add_executable( TARGET daml_ice_emul.x
                        SOURCES iceEmul.cc )

target_compile_features( daml_ice_emul.x  PUBLIC cxx_std_17)

target_link_libraries( daml_ice_emul.x PUBLIC NetCDF::NetCDF_CXX eckit daml
    "${Torch_ROOT}/lib/libtorch.so"
    "${Torch_ROOT}/lib/libtorch_cpu.so"
    "${Torch_ROOT}/lib/libc10.so"
    "${oops_ROOT}/lib/liboops.so")

# Salinity emulator
#ecbuild_add_executable( TARGET daml_salt_emul.x
#                        SOURCES saltEmul.cc)
#target_compile_features( daml_salt_emul.x  PUBLIC cxx_std_17)
#target_link_libraries( daml_salt_emul.x PUBLIC NetCDF::NetCDF_CXX eckit daml oops "${TORCH_LIBRARIES}")
