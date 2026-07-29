# Local build/upload changes (revert notes)

Record of one-off edits made while building and uploading
`ai.tar2.pose_kv_store_skill_v1`. Restore the "Previously" blocks to undo.

---

## 1. `src/sdk-ros/grpc_vendor/CMakeLists.txt`

**Workspace file actually used by the Docker build:**

`/home/rschnurr/ws_copy_of_copy_satya_robert/src/sdk-ros/grpc_vendor/CMakeLists.txt`

(`ws_aic/src/sdk-ros/grpc_vendor/CMakeLists.txt` was left alone; it differed
only by lacking `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.)

### Why

Cold Docker skill builds failed in `gz_transport_vendor` with:

```text
#error "Protobuf C++ gencode is built with an incompatible version of"
```

Cause: this file always vendored gRPC with `gRPC_PROTOBUF_PROVIDER=module`,
which installs protobuf **31.1.0** into the install tree. `gz_msgs` then
generated code against that, while compile of `gz_transport` picked up conda
protobuf **6.33.5** from the pixi skill env. Upstream `sdk-ros` `main` already
skips the vendored gRPC build when `find_package(gRPC)` succeeds; the pixi
skill-build env provides `libgrpc` / `gRPCConfig.cmake`.

### Previously (full file before this change)

```cmake
cmake_minimum_required(VERSION 3.10)
project(grpc_vendor CXX)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)
find_package(abseil_cpp_vendor REQUIRED)
find_package(Protobuf REQUIRED)
find_package(re2_vendor REQUIRED)
find_package(re2 REQUIRED)

ament_vendor(grpc_vendor
  VCS_URL https://github.com/grpc/grpc
  VCS_VERSION v1.74.0
  CMAKE_ARGS
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DABSL_BUILD_TESTING:BOOL=OFF
    -DABSL_ENABLE_INSTALL:BOOL=ON
    -DABSL_PROPAGATE_CXX_STD:BOOL=ON
    -DgRPC_INSTALL:BOOL=ON
    -DgRPC_ABSL_PROVIDER=package
    -DgRPC_RE2_PROVIDER=package
    -DgRPC_BUILD_TESTS:BOOL=OFF
    -DgRPC_BUILD_CSHARP_EXT:BOOL=OFF
    -DgRPC_BUILD_GRPC_CSHARP_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_NODE_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_PHP_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_NODE_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_OBJECTIVEC_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_RUBY_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=ON
    -DgRPC_PROTOBUF_PROVIDER:STRING=module
    -DPROTOBUF_ROOT_DIR=${PROTOBUF_ROOT_DIR}
  GLOBAL_HOOK
)

if (TARGET grpc_vendor)
  # Install the gRPC proto definitions so that we can use them elsewhere
  ExternalProject_Get_Property(grpc_vendor SOURCE_DIR)
  install(DIRECTORY "${SOURCE_DIR}/src/proto/grpc" # source directory
          DESTINATION "opt/${PROJECT_NAME}/share/grpc-proto/src/proto/" # target directory
          FILES_MATCHING # install only matched files
          PATTERN "*.proto" # select header files
  )
endif()

ament_export_dependencies(
  abseil_cpp_vendor
  re2_vendor
)

ament_package()
```

### Changed to (full file after this change)

```cmake
cmake_minimum_required(VERSION 3.10)
project(grpc_vendor CXX)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

# Prefer the pixi/conda gRPC when present. Building the vendored copy with
# gRPC_PROTOBUF_PROVIDER=module installs protobuf 31.1.0 into the install tree
# and then breaks gz_msgs/gz_transport against conda protobuf 6.33.5.
set(grpc_satisfied_by_system FALSE)
find_package(gRPC QUIET)
if(gRPC_FOUND)
  message(STATUS "Found gRPC, skipping build.")
  set(grpc_satisfied_by_system TRUE)
else()
  message(STATUS "gRPC not found, building.")
  set(grpc_satisfied_by_system FALSE)
endif()

if(NOT grpc_satisfied_by_system)
  find_package(abseil_cpp_vendor REQUIRED)
  find_package(Protobuf REQUIRED)
  find_package(re2_vendor REQUIRED)
  find_package(re2 REQUIRED)
endif()

ament_vendor(grpc_vendor
  VCS_URL https://github.com/grpc/grpc
  VCS_VERSION v1.74.0
  SATISFIED ${grpc_satisfied_by_system}
  CMAKE_ARGS
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DABSL_BUILD_TESTING:BOOL=OFF
    -DABSL_ENABLE_INSTALL:BOOL=ON
    -DABSL_PROPAGATE_CXX_STD:BOOL=ON
    -DgRPC_INSTALL:BOOL=ON
    -DgRPC_ABSL_PROVIDER=package
    -DgRPC_RE2_PROVIDER=package
    -DgRPC_BUILD_TESTS:BOOL=OFF
    -DgRPC_BUILD_CSHARP_EXT:BOOL=OFF
    -DgRPC_BUILD_GRPC_CSHARP_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_NODE_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_PHP_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_NODE_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_OBJECTIVEC_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_RUBY_PLUGIN:BOOL=OFF
    -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=ON
    -DgRPC_PROTOBUF_PROVIDER:STRING=module
    -DPROTOBUF_ROOT_DIR=${PROTOBUF_ROOT_DIR}
  GLOBAL_HOOK
)

if (TARGET grpc_vendor)
  # Install the gRPC proto definitions so that we can use them elsewhere
  ExternalProject_Get_Property(grpc_vendor SOURCE_DIR)
  install(DIRECTORY "${SOURCE_DIR}/src/proto/grpc" # source directory
          DESTINATION "opt/${PROJECT_NAME}/share/grpc-proto/src/proto/" # target directory
          FILES_MATCHING # install only matched files
          PATTERN "*.proto" # select header files
  )
endif()

ament_export_dependencies(
  abseil_cpp_vendor
  re2_vendor
)

ament_package()
```

Summary of the delta vs previously:

1. Probe for system/conda gRPC with `find_package(gRPC QUIET)`.
2. Only `find_package` abseil / Protobuf / re2 when gRPC is **not** found.
3. Pass `SATISFIED ${grpc_satisfied_by_system}` into `ament_vendor` so the
   vendored gRPC (and its bundled protobuf 31.1.0) is skipped when pixi already
   provides gRPC.
4. Kept the local `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` arg for the fallback
   vendored build path (upstream `main` dropped it; we kept it so a no-gRPC
   fallback still matches what this checkout used before).

Pattern taken from
https://github.com/intrinsic-ai/sdk-ros/blob/main/grpc_vendor/CMakeLists.txt
(as of 2026-07-27).

### How to revert

Replace
`/home/rschnurr/ws_copy_of_copy_satya_robert/src/sdk-ros/grpc_vendor/CMakeLists.txt`
with the "Previously" block above (or `git checkout -- grpc_vendor/CMakeLists.txt`
inside that sdk-ros checkout if the file was tracked and this was the only
edit).

---

## 1b. Same file — second attempt (2026-07-27 evening)

### Result of attempt 1

gRPC skip worked (`Found gRPC, skipping build`, `grpc_vendor` ~1.3s), but the
build then failed in `gz_msgs_vendor` linking `gz-msgs11_protoc_plugin`:

```text
undefined reference to absl::...DeallocateBackingArray...
libabsl_raw_hash_set.so: error adding symbols: DSO missing from command line
Failed <<< gz_msgs_vendor
```

So attempt 1 is **superseded** by attempt 2 below. Do not leave the
SATISFIED/skip version in place if you want the state that matches attempt 2.

### Why attempt 2

Keep building vendored gRPC (which got past `gz_msgs` on the original cold
build), but stop it from installing protobuf 31.1.0. Change only:

`gRPC_PROTOBUF_PROVIDER:STRING=module` → `gRPC_PROTOBUF_PROVIDER:STRING=package`

so gRPC links/uses the pixi/conda protobuf 6.33.5 that `gz_msgs` /
`gz_transport` already see.

### Previously (state after attempt 1 — the skip version)

The full "Changed to" file under §1 above (SATISFIED + `find_package(gRPC)`).

### Changed to (attempt 2 — current file)

Restored the original always-build structure from §1 "Previously", with the
single intentional delta:

```cmake
-    -DgRPC_PROTOBUF_PROVIDER:STRING=module
+    -DgRPC_PROTOBUF_PROVIDER:STRING=package
```

plus a short comment explaining why. Full file path:

`/home/rschnurr/ws_copy_of_copy_satya_robert/src/sdk-ros/grpc_vendor/CMakeLists.txt`

### How to revert to stock sdk-ros

Use the §1 "Previously" block (original `module` provider, no SATISFIED).

### How to revert only attempt 2 (restore attempt 1 skip)

Use the §1 "Changed to" block.

### Result of attempt 2

`gRPC_PROTOBUF_PROVIDER=package` failed while linking gRPC plugins against
conda protobuf 6.33.5:

```text
undefined reference to google::protobuf::io::Printer::PrintImpl(...)
Failed <<< grpc_vendor
```

grpc 1.74 expects its bundled protobuf 31.1.0 ABI; conda 6.33.5 is not a
drop-in. Attempt 2 is **superseded** by attempt 3.

### Attempt 3 (current) — restore stock grpc_vendor + strip after gRPC

1. Restored `grpc_vendor/CMakeLists.txt` to the §1 "Previously" stock file
   (`PROTOBUF_PROVIDER=module`, no SATISFIED skip).
2. Changed `flowstate/resources/Dockerfile.skill.cv` sdk-builder stage to a
   two-pass colcon build (see §3).

---

## 3. `flowstate/resources/Dockerfile.skill.cv` (attempt 3)

**Files:**
- `/home/rschnurr/satya/aic/flowstate/resources/Dockerfile.skill.cv`
- synced copy under
  `ws_copy_of_copy_satya_robert/src/aic/flowstate/resources/Dockerfile.skill.cv`

### Why

With stock `module` protobuf, the original cold build got past `gz_msgs` but
failed in `gz_transport` on the protobuf version `#error`, because gRPC's
installed `protoc`/headers (31.1.0) poisoned later packages. Skipping gRPC or
forcing `package` each failed differently (see §1b). Attempt 3 keeps gRPC's
bundled **libs** (needed to link/run libgrpc) but removes the **codegen
surface** before building the rest of the SDK.

### Previously (sdk-builder stage)

```dockerfile
FROM pixi-builder AS sdk-builder
COPY src/sdk-ros/ ./sdk-ros/
RUN pixi run --environment skill-build colcon build \
    --merge-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --event-handlers=console_direct+ \
    --packages-up-to intrinsic_sdk \
    --base-paths sdk-ros
```

### Changed to

```dockerfile
FROM pixi-builder AS sdk-builder
COPY src/sdk-ros/ ./sdk-ros/
# grpc_vendor builds with gRPC_PROTOBUF_PROVIDER=module and installs protobuf
# 31.1.0 into opt/grpc_vendor. Later gz_msgs/gz_transport then mix that with
# pixi's protobuf 6.33.5 and die on the C++ gencode version check. Build gRPC
# first (it needs its matching bundled protobuf libs at link/runtime), strip
# only the codegen/headers/cmake that would hijack later packages, then finish
# the SDK against the conda protobuf.
RUN pixi run --environment skill-build colcon build \
    --merge-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --event-handlers=console_direct+ \
    --packages-up-to grpc_vendor \
    --base-paths sdk-ros \
 && rm -rf /workspace/install/opt/grpc_vendor/include/google/protobuf \
 && rm -f /workspace/install/opt/grpc_vendor/bin/protoc \
          /workspace/install/opt/grpc_vendor/bin/protoc-* \
 && rm -rf /workspace/install/opt/grpc_vendor/lib/cmake/protobuf \
 && rm -f /workspace/install/opt/grpc_vendor/lib/pkgconfig/protobuf*.pc \
 && pixi run --environment skill-build colcon build \
    --merge-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --event-handlers=console_direct+ \
    --packages-up-to intrinsic_sdk \
    --base-paths sdk-ros
```

(Note: no trailing `\` after the final `--base-paths sdk-ros` line — a leftover
continuation backslash made Docker treat the next `FROM sdk-builder AS build`
as `FROM build:latest` and abort immediately.)

### How to revert

Restore the "Previously" sdk-builder stage above in both Dockerfile copies.
Also keep `grpc_vendor/CMakeLists.txt` as the §1 stock file (already restored
for attempt 3).

### Result of attempt 3

Two-pass strip still failed the same way on `gz_transport_vendor`, but the log
showed **`Built target gz-transport14`** (C++ library OK) before the pybind
target died on the protobuf `#error`. Attempts 1–3 were solving the wrong
layer of the problem.

### Attempt 4 (current) — skip gz-transport Python bindings

1. **Reverted** `Dockerfile.skill.cv` sdk-builder to the original single-pass
   colcon (stock "Previously" block in this section).
2. **Patched** `gz_transport_vendor/CMakeLists.txt` to pass
   `-DSKIP_PYBIND11=ON` (see §4). Stock `grpc_vendor` kept as in §1 Previously.

---

## 4. `src/sdk-ros/gz_transport_vendor/CMakeLists.txt` (attempt 4)

**File:**
`/home/rschnurr/ws_copy_of_copy_satya_robert/src/sdk-ros/gz_transport_vendor/CMakeLists.txt`

### Why

Cold builds always fail compiling
`python/.../_gz_transport_pybind11.cc` with the protobuf gencode/runtime
version `#error`. The C++ target `libgz-transport14.so` builds successfully.
`intrinsic_sdk_cmake` links `gz-transport14::gz-transport14` only; it does not
need the pybind module. Gazebo's own CMake supports `SKIP_PYBIND11`.

### Previously

```cmake
cmake_minimum_required(VERSION 3.10)
project(gz_transport_vendor CXX)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)
find_package(grpc_vendor REQUIRED)  # Provides protobuf
find_package(gz_cmake_vendor REQUIRED)
find_package(gz_math_vendor REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz_utils_vendor REQUIRED)

ament_vendor(gz_transport_vendor
  VCS_URL https://github.com/gazebosim/gz-transport
  # Target Intrinsic SDK version requires Gazebo Ionic packages.
  VCS_VERSION gz-transport14_14.1.0
  GLOBAL_HOOK
)

ament_package()
```

### Changed to

Same file, plus `CMAKE_ARGS -DSKIP_PYBIND11=ON` and a short comment. Full
current file is the one on disk at the path above.

### How to revert

Replace with the "Previously" block (or `git checkout -- gz_transport_vendor/CMakeLists.txt`
in that sdk-ros tree).

---

## 2. Staged skill sources (not an sdk-ros edit)

For the Docker build context, `aic_kv_store` and
`build_pose_kv_store_skill.sh` were rsynced from
`/home/rschnurr/satya/aic/flowstate/` into
`/home/rschnurr/ws_copy_of_copy_satya_robert/src/aic/flowstate/`.

That is packaging for build, not a semantic change to sdk-ros. The
authoritative skill sources remain under `satya/aic/flowstate/aic_kv_store/`.

### Active local patches for the next build (summary)

| Path | State |
|---|---|
| `sdk-ros/grpc_vendor/CMakeLists.txt` | stock (§1 Previously) |
| `sdk-ros/gz_transport_vendor/CMakeLists.txt` | `SKIP_PYBIND11=ON` (§4) |
| `flowstate/resources/Dockerfile.skill.cv` | stock single-pass sdk-builder |
| `flowstate/aic_kv_store/pose_keys.cc` | local `JoinKeySegments` (§5) |

---

## 5. `flowstate/aic_kv_store/pose_keys.cc` — no `KeyValueStore::MakeKey` on SDK v1.28

### Why

Build 5 got past the full SDK (`intrinsic_sdk` / `intrinsic_sdk_cmake`
finished). Compiling `aic_kv_store` then failed:

```text
pose_keys.cc:161: error: 'MakeKey' is not a member of 'intrinsic::KeyValueStore'
```

`MakeKey` exists on newer Intrinsic SDK tags (e.g. v1.33) but **not** on the
pinned `v1.28.20260223` this workspace builds. Slash-joined zenoh keys are
still valid on v1.28 (`ValidZenohKeyexpr`); only the helper is missing.

### Previously

```cpp
#include "intrinsic/platform/pubsub/kvstore.h"
// ...
return ::intrinsic::KeyValueStore::MakeKey(prefix, *slug);
// and
return ::intrinsic::KeyValueStore::MakeKey(prefix, *slug, absl::StrCat(index));
```

### Changed to

Dropped the `kvstore.h` include from `pose_keys.cc`. Added a file-local
`JoinKeySegments` that strips leading/trailing `/` per part and joins with
`/`, then call that from `MakePoseKey` instead of `KeyValueStore::MakeKey`.

### How to revert

Restore the `kvstore.h` include and the two `KeyValueStore::MakeKey` call
sites (only safe after bumping the SDK past the commit that added `MakeKey`).

---

## 6. `flowstate/scripts/build_pose_kv_store_skill.sh` — no duplicate proto desc

### Why

Docker image + smoke packaging succeeded. `inbuild skill manifest` then failed:

```text
proto: file appears multiple times: "intrinsic/math/proto/point.proto"
```

because both `pose_kv_store_skill_protos.desc` (already contains point/pose/
quaternion via imports) and `intrinsic_proto.desc` were passed. Same lesson as
`build_check_board_visibility_skill.sh`.

### Previously

```bash
docker cp .../${SKILL_NAME}_protos.desc ...
docker cp .../intrinsic_proto.desc ...
"${INBUILD_BIN}" skill manifest \
  --file_descriptor_sets "${OUTPUT_DIR}/${SKILL_NAME}_protos.desc,${OUTPUT_DIR}/intrinsic_proto.desc" \
  ...
```

### Changed to (attempt A — insufficient)

Copy only the skill `_protos.desc`, and pass that single set to
`inbuild skill manifest`. That avoided the duplicate, but
`inbuild skill bundle` then failed needing platform services:

```text
could not find service "intrinsic_proto.skills.Projector" in provided descriptors
```

### Changed to (kept — attempt B)

Copy both desc files, then **merge by proto file name** (skill first, then
`intrinsic_proto.desc`, skip names already seen) into
`${SKILL_NAME}_protos.merged.desc`. Pass that single merged set to
`inbuild skill manifest` / `skill bundle`. Uses AIC pixi `python` for
`google.protobuf` (system `python3` often lacks it).

### How to revert

Drop the merge step and restore either dual-desc (duplicates) or skill-only
desc (missing Projector).

---

## 7. Ignore Flowstate's default singular `pose` on indexed writes

### Why

Flowstate always materialises the singular `pose` parameter. First it looked
like all zeros; the UI actually defaults orientation `w` to `1` (identity at
origin). Either way `has_pose()` is true, so a NIC/SFP/SC write failed with
"the five nic poses belong in the poses list" even when only `poses` was filled.

### Changed to

Indexed writes (`NIC`/`SFP`/`SC`) **ignore** the singular `pose` field entirely
and only use `poses`. `PoseIsUiDefault` treats origin + (`w=0` or `w=1`) as
unset for the home-write path so a default message is not taken as a real home.

### How to revert

Restore the plain `params.has_pose()` rejection on indexed writes and drop
`PoseIsUiDefault`.

---
