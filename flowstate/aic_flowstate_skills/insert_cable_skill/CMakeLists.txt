intrinsic_sdk_generate_skill(
  SKILL_NAME insert_cable_skill
  SOURCES insert_cable_skill.proto
  MANIFEST insert_cable_skill.manifest.textproto
  PROTOS_TARGET insert_cable_skill_protos
)

intrinsic_sdk_generate_skill_manifest_pbbin(
  TARGET insert_cable_skill_manifest
  MANIFEST_TEXTPROTO insert_cable_skill.manifest.textproto
  MANIFEST_PBBIN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill.manifest.pbbin"
)

intrinsic_sdk_generate_skill_main_cc(
  MANIFEST_PBBIN "${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill.manifest.pbbin"
  CONFIG_PBBIN "${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill_config.pbbin"
  HEADER_FILES insert_cable_skill.h
  MAIN_FILE_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill_main.cc
)

install(
  FILES
    insert_cable_skill.manifest.textproto
    "${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill.manifest.pbbin"
    "${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill_config.pbbin"
    "${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill_protos.desc"
  DESTINATION share/${PROJECT_NAME}
)

add_executable(insert_cable_skill_main
  insert_cable_skill.cc
  insert_cable_action_client.cc
  ${CMAKE_CURRENT_BINARY_DIR}/insert_cable_skill_main.cc
)

target_include_directories(insert_cable_skill_main
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(insert_cable_skill_main
  intrinsic_sdk_cmake::intrinsic_sdk_cmake
  insert_cable_skill_protos
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  ${aic_task_interfaces_TARGETS}
)

add_dependencies(insert_cable_skill_main insert_cable_skill_config)

install(
  TARGETS insert_cable_skill_main
  DESTINATION lib/${PROJECT_NAME}
)
