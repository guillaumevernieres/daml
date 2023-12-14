add_executable( daml_ice_emul.x ice_emul.cc)
set_property(TARGET daml_ice_emul.x PROPERTY CXX_STANDARD 17)

target_link_libraries( daml_ice_emul.x PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries( daml_ice_emul.x PRIVATE oops eckit ioda)

install(TARGETS daml_ice_emul.x RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
