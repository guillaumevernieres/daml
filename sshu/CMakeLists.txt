add_executable( daml_sshu.x daml_sshu.cc)
set_property(TARGET daml_sshu.x PROPERTY CXX_STANDARD 17)

target_link_libraries( daml_sshu.x PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries( daml_sshu.x PRIVATE oops)

install(TARGETS daml_sshu.x RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
