include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

list( APPEND daml_src_files
  daml/Base/BaseEmul.h
  daml/IceEmul/IceEmul.h
  daml/IceEmul/IceNet.h
  daml/SaltEmul/SaltEmul.h
  daml/SaltEmul/SaltNet.h
  )
set_source_files_properties(${daml_src_files} PROPERTIES LANGUAGE CXX)
add_library(${PROJECT_NAME} ${daml_src_files})
target_link_libraries( ${PROJECT_NAME} PUBLIC NetCDF::NetCDF_CXX oops eckit atlas "${TORCH_LIBRARIES}")
set_target_properties( ${PROJECT_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(test)
