# ------------------------------------------------------------------------- #
# Application for all the components needed for the GDAS system             #
# ------------------------------------------------------------------------- #

# Check for minimim cmake requirement
cmake_minimum_required( VERSION 3.20 FATAL_ERROR )

project(DAML VERSION 1.0.0 LANGUAGES C CXX Fortran )

find_package(Torch REQUIRED)
find_package(oops REQUIRED)

# Include sshu.
add_subdirectory(sshu)
