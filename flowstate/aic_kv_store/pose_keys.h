/*
 * Copyright (C) 2026 TAR-2
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLOWSTATE_AIC_KV_STORE_POSE_KEYS_H_
#define FLOWSTATE_AIC_KV_STORE_POSE_KEYS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "intrinsic/math/proto/pose.pb.h"
#include "pose_kv_store_skill.pb.h"

namespace aic_kv {

using ObjectType = ::ai::tar2::PoseKvStoreSkillParams::ObjectType;

// Every indexed type has exactly one pose per official label 0..4.
inline constexpr int kPosesPerType = 5;

// The index of a type that holds a single pose, such as home.
inline constexpr int kNoIndex = -1;

inline constexpr absl::string_view kDefaultKeyPrefix = "aic/phase1";

struct ResolvedTarget {
  ObjectType type;
  // kNoIndex when the name is an index-less type such as "home".
  int index;
};

// "sfp", "nic", "sc" or "home".
absl::StatusOr<std::string> ObjectTypeSlug(ObjectType type);

// False for the types that hold one pose under a bare key instead of five
// poses under numbered keys.
bool TypeHasIndex(ObjectType type);

// Resolves a module name from a trial's process inputs, such as
// "nic_card_mount_3", "sc_port_1", "sfp_mount_2" or "home", into its type and
// its official index. Matching is deliberately forgiving about decoration
// around the name (case, hyphens, a leading path, a "rail" infix) but strict
// about the two things that carry meaning: the type word and, for the indexed
// types, the trailing index.
absl::StatusOr<ResolvedTarget> ParseTargetName(absl::string_view target_name);

// Rejects a prefix that zenoh cannot store under, so a bad prefix fails on the
// first key instead of writing part of a set.
absl::Status ValidateKeyPrefix(absl::string_view key_prefix);

// <key_prefix>/<slug>/<index>, with an empty prefix meaning kDefaultKeyPrefix.
// A type with no index gets <key_prefix>/<slug> and ignores the index.
absl::StatusOr<std::string> MakePoseKey(absl::string_view key_prefix,
                                        ObjectType type, int index);

// Rejects poses that cannot be acted on downstream: a non-finite component, or
// an orientation whose quaternion is unset or degenerate. An all-zero
// quaternion reaching Move Robot is a crash, not a bad move, so it is worth
// catching on the way into and out of the store.
absl::Status ValidatePose(const ::intrinsic_proto::Pose& pose,
                          absl::string_view what);

// True when the pose looks like an untouched Flowstate default rather than a
// wired value: origin with either an all-zero quaternion or identity (w=1).
// A real survey pose at exactly the origin with identity orientation is also
// treated as default; use the poses list for that edge case.
bool PoseIsUiDefault(const ::intrinsic_proto::Pose& pose);

}  // namespace aic_kv

#endif  // FLOWSTATE_AIC_KV_STORE_POSE_KEYS_H_
