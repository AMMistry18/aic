cmake_minimum_required(VERSION 3.8)
project(aic_kv_store)

# The SDK's key-value store templates use C++20 constrained members.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(intrinsic_sdk_cmake REQUIRED)

set(SKILL_NAME pose_kv_store_skill)

# Descriptor set consumed by inbuild and Flowstate, plus the C++ bindings for
# the params and result messages.
intrinsic_sdk_protobuf_generate(
  NAME ${SKILL_NAME}
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.proto"
  TARGET ${SKILL_NAME}_protos
  DESCRIPTOR_SET_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_protos.desc"
)

# Runtime config consumed by the Intrinsic skill service.
intrinsic_sdk_generate_skill_config(
  TARGET ${SKILL_NAME}_config
  SKILL_NAME ${SKILL_NAME}
  MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.manifest.textproto"
  PROTO_DESCRIPTOR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_protos.desc"
  SKILL_CONFIG_FILE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_config.pbbin"
)

intrinsic_sdk_generate_skill_main_cc(
  MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.manifest.textproto"
  HEADER_FILES ${SKILL_NAME}.h
  MAIN_FILE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_main.cc"
)

# Key scheme and name grammar, kept separate so they can be unit tested without
# a key-value store or a running skill service.
add_library(pose_keys STATIC pose_keys.cc)
target_include_directories(pose_keys PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pose_keys
  PUBLIC
    ${SKILL_NAME}_protos
    intrinsic_sdk_cmake::intrinsic_sdk_cmake
)
set_property(TARGET pose_keys PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(${SKILL_NAME}_main
  ${SKILL_NAME}.cc
  "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_main.cc"
)

target_include_directories(${SKILL_NAME}_main
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${SKILL_NAME}_main
  intrinsic_sdk_cmake::intrinsic_sdk_cmake
  ${SKILL_NAME}_protos
  pose_keys
)

add_dependencies(${SKILL_NAME}_main ${SKILL_NAME}_config)
add_dependencies(${SKILL_NAME}_main ${SKILL_NAME}_protos)

install(
  TARGETS ${SKILL_NAME}_main
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/${SKILL_NAME}.manifest.textproto"
    "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_config.pbbin"
    "${CMAKE_CURRENT_BINARY_DIR}/${SKILL_NAME}_protos.desc"
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # Skipped rather than required, so a skill image build never fails on a
  # missing test dependency.
  find_package(ament_cmake_gtest QUIET)
  if(ament_cmake_gtest_FOUND)
    ament_add_gtest(test_pose_keys test/test_pose_keys.cc)
    target_link_libraries(test_pose_keys pose_keys)
  endif()
endif()

ament_package()
