include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

list( APPEND daml_src_files
  daml/Base/BaseEmul.h
  daml/IceEmul/IceEmul.h
  daml/IceEmul/IceNet.h
  daml/SaltEmul/SaltEmul.h
  daml/SaltEmul/SaltNet.h
  )

ecbuild_add_library( TARGET   daml
                     SOURCES ${daml_src_files}
                     INSTALL_HEADERS LISTED
                     LINKER_LANGUAGE CXX
                    )

#include_directories("${Torch_ROOT}/include")
#include_directories("${oops_ROOT}/oops/include/oops")
#include_directories("/work/noaa/da/gvernier/sandboxes/global-workflow/sorc/gdas.cd/sorc/oops/src")

#set_source_files_properties(${daml_src_files} PROPERTIES LANGUAGE CXX)
#add_library(${PROJECT_NAME} ${daml_src_files})
target_link_libraries( daml PUBLIC NetCDF::NetCDF_CXX )
if (oops_FOUND)
    target_link_libraries( daml PUBLIC oops )
else()
    if(EXISTS "${oops_ROOT}/lib/liboops.so")
      target_link_libraries(daml PUBLIC "${oops_ROOT}/lib/liboops.so")
    else()
      message(FATAL_ERROR "Oops library not found at ${oops_ROOT}/lib/liboops.so!")
    endif()
endif()

target_link_libraries( daml PUBLIC eckit )
target_link_libraries( daml PUBLIC atlas )
target_link_libraries( daml PUBLIC
    "${Torch_ROOT}/lib/libtorch.so"          # For the Torch core library
    "${Torch_ROOT}/lib/libtorch_cpu.so"      # For the CPU backend
    "${Torch_ROOT}/lib/libc10.so"            # Core library
    )
#target_link_libraries( daml PUBLIC "${TORCH_LIBRARIES}" )
set_target_properties( daml PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(test)
