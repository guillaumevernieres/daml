# ------------------------------------------------------------------------- #
# DAML Applications                                                         #
# ------------------------------------------------------------------------- #


cmake_minimum_required( VERSION 3.20 FATAL_ERROR )
find_package(ecbuild 3.5 REQUIRED)
#project(daml VERSION 1.0.0 LANGUAGES CXX )
project(daml VERSION 1.0.0 LANGUAGES CXX )

include( ecbuild_system NO_POLICY_SCOPE )
ecbuild_declare_project()

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include( daml_compiler_flags )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(eckit REQUIRED)
find_package(NetCDF REQUIRED COMPONENTS CXX)
find_package(atlas REQUIRED)
#find_package(Torch)
#if (NOT Torch_FOUND)
    # Temporary fix until Torch is properly installed as a cmake package
    # Ex. on Hercules:
    # Torch_ROOT=/work/noaa/epic/role-epic/spack-stack/hercules/spack-stack-1.7.0/envs/ue-gcc/install/gcc/12.2.0/py-torch-2.1.2-bgf7llx/lib/python3.10/site-packages/torch/
    include_directories("${Torch_ROOT}/include")
    include_directories("${Torch_ROOT}/include/torch/csrc/api/include")
#endif()
find_package(oops)
if (NOT oops_FOUND)
    # A bit of a hack but useful for now
    # Ex. on Hercules:
    # oops_ROOT=/work/noaa/da/gvernier/sandboxes/global-workflow/sorc/gdas.cd/build.gnu/
    include_directories("${oops_ROOT}/../sorc/oops/src")
endif()

include_directories( ${NETCDF_INCLUDE_DIRS} )

add_subdirectory(src)
add_subdirectory(app)
