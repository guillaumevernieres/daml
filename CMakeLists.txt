# ------------------------------------------------------------------------- #
# DAML Applications                                                         #
# ------------------------------------------------------------------------- #

# Check for minimim cmake requirement
cmake_minimum_required( VERSION 3.20 FATAL_ERROR )

project(DAML VERSION 1.0.0 LANGUAGES C CXX Fortran)

include(GNUInstallDirs)
enable_testing()

find_package(Torch REQUIRED)
find_package( eckit  1.23.0   REQUIRED)
find_package( fckit  0.10.1   REQUIRED)
find_package( atlas  0.35.0   REQUIRED)
find_package(oops REQUIRED)

# Include sshu.
add_subdirectory(sshu)

# Testing
add_subdirectory(test)
