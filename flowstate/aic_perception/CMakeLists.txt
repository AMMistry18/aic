cmake_minimum_required(VERSION 3.8)
project(aic_perception)

find_package(ament_cmake REQUIRED)
find_package(aic_control_interfaces REQUIRED)
find_package(intrinsic_sdk_cmake REQUIRED)

set(SKILL_NAME check_board_visibility_skill)

# Descriptor set consumed by inbuild and Flowstate.
intrinsic_sdk_protobuf_generate(
  NAME ${SKILL_NAME}
  SOURCES ${SKILL_NAME}.proto
  TARGET ${SKILL_NAME}_protos
  DESCRIPTOR_SET_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_protos.desc"
)

# Python bindings for the params and result messages.
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_pb2.py"
  COMMAND protoc -I${CMAKE_CURRENT_SOURCE_DIR}
          --python_out=${CMAKE_CURRENT_BINARY_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.proto
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.proto
)
add_custom_target(generate_python_proto_${SKILL_NAME} ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_pb2.py")

# Runtime config consumed by the Intrinsic Python skill service.
intrinsic_sdk_generate_skill_config(
  TARGET ${SKILL_NAME}_skill_config
  SKILL_NAME ${SKILL_NAME}
  MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.manifest.textproto"
  PROTO_DESCRIPTOR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_protos.desc"
  SKILL_CONFIG_FILE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_config.pbbin"
)

add_custom_target(force_manifest_generation_${SKILL_NAME} ALL)
add_dependencies(force_manifest_generation_${SKILL_NAME}
  ${SKILL_NAME}_protos
  generate_python_proto_${SKILL_NAME}
  ${SKILL_NAME}_skill_config
)

install(
  FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.manifest.textproto"
    "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_config.pbbin"
    "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_protos.desc"
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  ${SKILL_NAME}.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ${SKILL_NAME}_main
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_pb2.py"
    "${SKILL_NAME}.py"
  DESTINATION "lib/python3.12/site-packages/${PROJECT_NAME}"
)

install(
  DIRECTORY aic_perception/
  DESTINATION "lib/python3.12/site-packages/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.py"
)

ament_package()
