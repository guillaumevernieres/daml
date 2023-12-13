# Add the test
add_test(NAME daml_sshu
  COMMAND ${PROJECT_BINARY_DIR}/sshu/daml_sshu.x ${PROJECT_SOURCE_DIR}/test/daml_sshu.yaml
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/)
