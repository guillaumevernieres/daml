# ------------------------------------------------------------------------- #
# DAML Applications                                                         #
# ------------------------------------------------------------------------- #

project(daml VERSION 1.0.0 LANGUAGES CXX )
find_package(ecbuild 3.3.2 REQUIRED)
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

find_package(Torch REQUIRED)
find_package(eckit REQUIRED)
find_package(NetCDF REQUIRED COMPONENTS CXX)
find_package(oops REQUIRED)

include_directories( ${NETCDF_INCLUDE_DIRS} )

add_subdirectory(src)
add_subdirectory(app)
