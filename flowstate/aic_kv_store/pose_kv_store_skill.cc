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

#include "pose_kv_store_skill.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "intrinsic/math/proto/pose.pb.h"
#include "intrinsic/util/status/status_macros.h"
#include "pose_keys.h"

namespace {

using Params = ::ai::tar2::PoseKvStoreSkillParams;
using Result = ::ai::tar2::PoseKvStoreSkillResult;

constexpr double kDefaultTimeoutSeconds = 10.0;

}  // namespace

std::unique_ptr<intrinsic::skills::SkillInterface>
PoseKvStoreSkill::CreateSkill() {
  return std::make_unique<PoseKvStoreSkill>();
}

absl::StatusOr<intrinsic_proto::skills::Footprint>
PoseKvStoreSkill::GetFootprint(
    const intrinsic::skills::GetFootprintRequest& /*request*/,
    intrinsic::skills::GetFootprintContext& /*context*/) const {
  return intrinsic_proto::skills::Footprint();
}

absl::StatusOr<std::unique_ptr<google::protobuf::Message>>
PoseKvStoreSkill::Preview(
    const intrinsic::skills::PreviewRequest& /*request*/,
    intrinsic::skills::PreviewContext& /*context*/) {
  return absl::UnimplementedError("Preview not supported for this skill");
}

absl::StatusOr<intrinsic::KeyValueStore*> PoseKvStoreSkill::Store() {
  absl::MutexLock lock(&mutex_);
  if (!store_.has_value()) {
    if (!pubsub_.has_value()) {
      pubsub_.emplace();
    }
    INTR_ASSIGN_OR_RETURN(
        intrinsic::KeyValueStore store, pubsub_->KeyValueStore(),
        _ << "cannot reach the key-value store from this skill container");
    store_.emplace(std::move(store));
  }
  // Set once and never replaced, so the pointer outlives the lock.
  return &*store_;
}

absl::StatusOr<std::unique_ptr<google::protobuf::Message>>
PoseKvStoreSkill::Execute(const intrinsic::skills::ExecuteRequest& request,
                          intrinsic::skills::ExecuteContext& /*context*/) {
  INTR_ASSIGN_OR_RETURN(const Params params, request.params<Params>());
  INTR_RETURN_IF_ERROR(aic_kv::ValidateKeyPrefix(params.key_prefix()));

  switch (params.mode()) {
    case Params::MODE_WRITE:
      return WritePoses(params);
    case Params::MODE_READ:
      return ReadPose(params);
    default:
      return absl::InvalidArgumentError(
          "mode must be MODE_WRITE (store the poses of one type) or MODE_READ "
          "(return one pose)");
  }
}

absl::StatusOr<std::unique_ptr<google::protobuf::Message>>
PoseKvStoreSkill::WritePoses(const Params& params) {
  INTR_ASSIGN_OR_RETURN(const std::string slug,
                        aic_kv::ObjectTypeSlug(params.object_type()));

  if (!aic_kv::TypeHasIndex(params.object_type())) {
    return WriteSinglePose(params, slug);
  }
  // Singular `pose` is home-only. Flowstate always materialises it (often as
  // origin + identity w=1), so indexed writes ignore it entirely and only use
  // the poses list.

  if (params.poses_size() != aic_kv::kPosesPerType) {
    return absl::InvalidArgumentError(absl::StrCat(
        "write mode needs exactly ", aic_kv::kPosesPerType, " ", slug,
        " poses, one per official label, but got ", params.poses_size(),
        "; a partially filled type makes every later read by index "
        "meaningless"));
  }
  for (int i = 0; i < params.poses_size(); ++i) {
    INTR_RETURN_IF_ERROR(aic_kv::ValidatePose(
        params.poses(i), absl::StrCat(slug, " pose ", i)));
  }

  INTR_ASSIGN_OR_RETURN(intrinsic::KeyValueStore * store, Store());

  auto result = std::make_unique<Result>();
  std::vector<std::string> keys;
  keys.reserve(params.poses_size());
  for (int i = 0; i < params.poses_size(); ++i) {
    INTR_ASSIGN_OR_RETURN(
        const std::string key,
        aic_kv::MakePoseKey(params.key_prefix(), params.object_type(), i));
    if (absl::Status status =
            store->Set(key, params.poses(i), /*high_consistency=*/true);
        !status.ok()) {
      const std::string written =
          keys.empty() ? std::string("none of the earlier keys were written")
                       : absl::StrCat("already written: ",
                                      absl::StrJoin(keys, ", "));
      return absl::Status(
          status.code(), absl::StrCat("failed to write '", key, "' (", written,
                                      "): ", status.message()));
    }
    keys.push_back(key);
    result->add_keys_written(key);
  }

  result->set_success(true);
  result->set_resolved_type(slug);
  result->set_message(absl::StrCat("stored ", keys.size(), " ", slug,
                                   " poses at ", absl::StrJoin(keys, ", ")));
  LOG(INFO) << result->message();
  return result;
}

absl::StatusOr<std::unique_ptr<google::protobuf::Message>>
PoseKvStoreSkill::WriteSinglePose(const Params& params,
                                  const std::string& slug) {
  const intrinsic_proto::Pose* pose = nullptr;
  const bool pose_field_set =
      params.has_pose() && !aic_kv::PoseIsUiDefault(params.pose());
  if (pose_field_set) {
    if (params.poses_size() != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "the ", slug, " key holds one pose, but both the pose field and ",
          params.poses_size(), " entries in poses were given"));
    }
    pose = &params.pose();
  } else if (params.poses_size() == 1) {
    pose = &params.poses(0);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("write mode for ", slug,
                     " needs its one pose in the pose field, but that field is "
                     "unset and poses holds ",
                     params.poses_size(), " entries"));
  }
  INTR_RETURN_IF_ERROR(
      aic_kv::ValidatePose(*pose, absl::StrCat(slug, " pose")));

  INTR_ASSIGN_OR_RETURN(
      const std::string key,
      aic_kv::MakePoseKey(params.key_prefix(), params.object_type(),
                          aic_kv::kNoIndex));
  INTR_ASSIGN_OR_RETURN(intrinsic::KeyValueStore * store, Store());
  if (absl::Status status = store->Set(key, *pose, /*high_consistency=*/true);
      !status.ok()) {
    return absl::Status(status.code(), absl::StrCat("failed to write '", key,
                                                    "': ", status.message()));
  }

  auto result = std::make_unique<Result>();
  result->set_success(true);
  result->set_resolved_type(slug);
  result->set_resolved_index(aic_kv::kNoIndex);
  result->add_keys_written(key);
  result->set_message(absl::StrCat("stored the ", slug, " pose at ", key));
  LOG(INFO) << result->message();
  return result;
}

absl::StatusOr<std::unique_ptr<google::protobuf::Message>>
PoseKvStoreSkill::ReadPose(const Params& params) {
  aic_kv::ObjectType type = params.object_type();
  int index = aic_kv::kNoIndex;
  std::string index_origin;
  std::string named_type;

  if (!params.target_name().empty()) {
    INTR_ASSIGN_OR_RETURN(const aic_kv::ResolvedTarget target,
                          aic_kv::ParseTargetName(params.target_name()));
    index = target.index;
    index_origin =
        absl::StrCat("index came from target_name '", params.target_name(),
                     "'");
    if (type == Params::OBJECT_TYPE_UNSPECIFIED) {
      type = target.type;
    } else if (type != target.type) {
      // This is how a trial picks its cable: the NIC card number of the trial
      // stands in for an SFP index. Any unused module will do, so borrowing the
      // index of another type is intended rather than a mismatch.
      INTR_ASSIGN_OR_RETURN(named_type, aic_kv::ObjectTypeSlug(target.type));
    }
  } else {
    if (type == Params::OBJECT_TYPE_UNSPECIFIED) {
      return absl::InvalidArgumentError(
          "read mode needs target_name (e.g. nic_card_mount_3 or home), or "
          "object_type together with index");
    }
    index = params.index();
    index_origin = "index came from the index parameter";
  }

  INTR_ASSIGN_OR_RETURN(const std::string slug, aic_kv::ObjectTypeSlug(type));
  if (!aic_kv::TypeHasIndex(type)) {
    // One pose under one key, so whatever index the caller supplied is moot.
    index = aic_kv::kNoIndex;
    index_origin = absl::StrCat(slug, " is a single key with no index");
  } else if (index == aic_kv::kNoIndex) {
    return absl::InvalidArgumentError(
        absl::StrCat("target_name '", params.target_name(),
                     "' carries no index, so it cannot select one of the five ",
                     slug, " poses"));
  } else if (!named_type.empty()) {
    absl::StrAppend(&index_origin, ", a ", named_type,
                    " name whose index this read reuses for ", slug);
  }
  INTR_ASSIGN_OR_RETURN(const std::string key,
                        aic_kv::MakePoseKey(params.key_prefix(), type, index));

  if (params.timeout_seconds() < 0.0) {
    return absl::InvalidArgumentError("timeout_seconds must not be negative");
  }
  const double timeout_seconds = params.timeout_seconds() > 0.0
                                     ? params.timeout_seconds()
                                     : kDefaultTimeoutSeconds;

  INTR_ASSIGN_OR_RETURN(intrinsic::KeyValueStore * store, Store());
  absl::StatusOr<intrinsic_proto::Pose> pose =
      store->Get<intrinsic_proto::Pose>(key, absl::Seconds(timeout_seconds));
  if (!pose.ok()) {
    return absl::Status(
        pose.status().code(),
        absl::StrCat("no ", slug, " pose readable at '", key, "' within ",
                     timeout_seconds, "s: ", pose.status().message(),
                     "; run the write for ", slug,
                     " once in Participant Initialize before any trial reads "
                     "it"));
  }
  INTR_RETURN_IF_ERROR(aic_kv::ValidatePose(*pose, absl::StrCat("'", key, "'")));

  auto result = std::make_unique<Result>();
  result->set_success(true);
  result->set_resolved_type(slug);
  result->set_resolved_index(index);
  result->set_key(key);
  *result->mutable_pose() = *std::move(pose);
  const std::string what = aic_kv::TypeHasIndex(type)
                               ? absl::StrCat(slug, " pose ", index)
                               : absl::StrCat("the ", slug, " pose");
  result->set_message(
      absl::StrCat("read ", what, " from '", key, "'; ", index_origin));
  LOG(INFO) << result->message();
  return result;
}
