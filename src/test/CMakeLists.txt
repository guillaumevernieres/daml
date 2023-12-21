ecbuild_add_test( TARGET test_daml_coding_norms
                  TYPE SCRIPT
                  COMMAND ${CMAKE_BINARY_DIR}/bin/cpplint.py
                  ARGS --quiet --recursive ${CMAKE_CURRENT_SOURCE_DIR}/../
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# TODO(G): Add test nc file
#add_test(NAME daml_ice
#  COMMAND ${PROJECT_BINARY_DIR}/../bin/daml_ice_emul.x ${PROJECT_SOURCE_DIR}/test/daml_sshu.yaml
#  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/)
