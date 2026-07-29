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

#include "pose_keys.h"

#include <cmath>
#include <initializer_list>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"

namespace aic_kv {
namespace {

using Params = ::ai::tar2::PoseKvStoreSkillParams;

// Characters zenoh reserves for key expressions.
constexpr absl::string_view kForbiddenKeyChars = "*?#[]";

bool IsFinite(double value) { return std::isfinite(value); }

// Local stand-in for KeyValueStore::MakeKey, which this SDK pin (v1.28) does
// not yet expose. Matches the later SDK helper: strip leading/trailing '/'
// from each part and join with '/'.
std::string JoinKeySegments(
    std::initializer_list<absl::string_view> parts) {
  std::string result;
  for (absl::string_view part : parts) {
    while (!part.empty() && part.front() == '/') {
      part.remove_prefix(1);
    }
    while (!part.empty() && part.back() == '/') {
      part.remove_suffix(1);
    }
    if (part.empty()) {
      continue;
    }
    if (!result.empty()) {
      result.push_back('/');
    }
    result.append(part.data(), part.size());
  }
  return result;
}

}  // namespace

absl::StatusOr<std::string> ObjectTypeSlug(ObjectType type) {
  switch (type) {
    case Params::OBJECT_TYPE_SFP:
      return "sfp";
    case Params::OBJECT_TYPE_NIC:
      return "nic";
    case Params::OBJECT_TYPE_SC:
      return "sc";
    case Params::OBJECT_TYPE_HOME:
      return "home";
    default:
      return absl::InvalidArgumentError(
          "object_type must be OBJECT_TYPE_SFP, OBJECT_TYPE_NIC, "
          "OBJECT_TYPE_SC or OBJECT_TYPE_HOME");
  }
}

bool TypeHasIndex(ObjectType type) {
  return type != Params::OBJECT_TYPE_HOME;
}

absl::StatusOr<ResolvedTarget> ParseTargetName(absl::string_view target_name) {
  std::string name =
      absl::AsciiStrToLower(absl::StripAsciiWhitespace(target_name));
  const size_t last_slash = name.find_last_of('/');
  if (last_slash != std::string::npos) {
    name = name.substr(last_slash + 1);
  }
  absl::StrReplaceAll({{"-", "_"}, {".", "_"}, {" ", "_"}}, &name);
  if (name.empty()) {
    return absl::InvalidArgumentError(
        "target_name is empty; pass the module name from this trial's process "
        "inputs, e.g. nic_card_mount_3");
  }

  // Home is a single pose, so its name carries no index and is matched before
  // the trailing digits are looked for.
  if (absl::StrContains(name, "home")) {
    return ResolvedTarget{Params::OBJECT_TYPE_HOME, kNoIndex};
  }

  size_t index_begin = name.size();
  while (index_begin > 0 && absl::ascii_isdigit(name[index_begin - 1])) {
    --index_begin;
  }
  if (index_begin == name.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "target_name '", target_name,
        "' must end in the module index, e.g. nic_card_mount_3, sc_port_1 or "
        "sfp_mount_2"));
  }

  int index = 0;
  if (!absl::SimpleAtoi(absl::string_view(name).substr(index_begin), &index)) {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot read a module index out of '", target_name, "'"));
  }
  if (index < 0 || index >= kPosesPerType) {
    return absl::OutOfRangeError(
        absl::StrCat("target_name '", target_name, "' resolves to index ",
                     index, ", outside the 0..", kPosesPerType - 1,
                     " labels this store holds"));
  }

  const absl::string_view base =
      absl::string_view(name).substr(0, index_begin);
  ObjectType type;
  if (absl::StrContains(base, "nic")) {
    type = Params::OBJECT_TYPE_NIC;
  } else if (absl::StrContains(base, "sfp")) {
    type = Params::OBJECT_TYPE_SFP;
  } else if (absl::StrContains(base, "sc")) {
    type = Params::OBJECT_TYPE_SC;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "target_name '", target_name,
        "' does not say whether it is an sfp, nic or sc module; expected a "
        "name like sfp_mount_2, nic_card_mount_3, sc_port_1 or home"));
  }

  return ResolvedTarget{type, index};
}

absl::Status ValidateKeyPrefix(absl::string_view key_prefix) {
  if (key_prefix.empty()) {
    return absl::OkStatus();
  }
  if (key_prefix.find_first_of(kForbiddenKeyChars) != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        absl::StrCat("key_prefix '", key_prefix,
                     "' must not contain any of ", kForbiddenKeyChars));
  }
  if (absl::StrContains(key_prefix, "//")) {
    return absl::InvalidArgumentError(absl::StrCat(
        "key_prefix '", key_prefix, "' must not contain an empty segment"));
  }
  for (const char c : key_prefix) {
    if (absl::ascii_isspace(c)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "key_prefix '", key_prefix, "' must not contain whitespace"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> MakePoseKey(absl::string_view key_prefix,
                                        ObjectType type, int index) {
  if (absl::Status status = ValidateKeyPrefix(key_prefix); !status.ok()) {
    return status;
  }
  const absl::StatusOr<std::string> slug = ObjectTypeSlug(type);
  if (!slug.ok()) {
    return slug.status();
  }
  const absl::string_view prefix =
      key_prefix.empty() ? kDefaultKeyPrefix : key_prefix;
  if (!TypeHasIndex(type)) {
    return JoinKeySegments({prefix, *slug});
  }
  if (index < 0 || index >= kPosesPerType) {
    return absl::OutOfRangeError(absl::StrCat("index ", index, " is outside 0..",
                                              kPosesPerType - 1));
  }
  return JoinKeySegments({prefix, *slug, absl::StrCat(index)});
}

absl::Status ValidatePose(const ::intrinsic_proto::Pose& pose,
                          absl::string_view what) {
  if (!pose.has_position() || !pose.has_orientation()) {
    return absl::InvalidArgumentError(
        absl::StrCat(what, " has no position or no orientation set"));
  }
  const auto& p = pose.position();
  const auto& q = pose.orientation();
  if (!IsFinite(p.x()) || !IsFinite(p.y()) || !IsFinite(p.z()) ||
      !IsFinite(q.x()) || !IsFinite(q.y()) || !IsFinite(q.z()) ||
      !IsFinite(q.w())) {
    return absl::InvalidArgumentError(
        absl::StrCat(what, " has a non-finite component"));
  }
  const double norm =
      std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w());
  if (norm < 0.5 || norm > 2.0) {
    return absl::InvalidArgumentError(absl::StrCat(
        what, " has a degenerate orientation quaternion (norm ", norm,
        "); an unset or zero quaternion crashes Move Robot downstream"));
  }
  return absl::OkStatus();
}

bool PoseIsUiDefault(const ::intrinsic_proto::Pose& pose) {
  const auto& p = pose.position();
  const auto& q = pose.orientation();
  if (p.x() != 0.0 || p.y() != 0.0 || p.z() != 0.0) {
    return false;
  }
  if (q.x() != 0.0 || q.y() != 0.0 || q.z() != 0.0) {
    return false;
  }
  // Proto unset (w=0) or Flowstate identity default (w=1).
  return q.w() == 0.0 || q.w() == 1.0;
}

}  // namespace aic_kv
