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

#ifndef FLOWSTATE_AIC_KV_STORE_POSE_KV_STORE_SKILL_H_
#define FLOWSTATE_AIC_KV_STORE_POSE_KV_STORE_SKILL_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/message.h"
#include "intrinsic/platform/pubsub/kvstore.h"
#include "intrinsic/platform/pubsub/pubsub.h"
#include "intrinsic/skills/cc/skill_interface.h"
#include "pose_kv_store_skill.pb.h"

// Writes the five labelled module poses of one type, or the single home pose,
// into the key-value store, and reads one of them back by the name the trial
// already uses.
class PoseKvStoreSkill : public intrinsic::skills::SkillInterface {
 public:
  PoseKvStoreSkill() = default;
  ~PoseKvStoreSkill() override = default;

  static std::unique_ptr<intrinsic::skills::SkillInterface> CreateSkill();

  // The skill touches no equipment and no world objects, so it does not need
  // the default exclusive lock on the workcell.
  absl::StatusOr<intrinsic_proto::skills::Footprint> GetFootprint(
      const intrinsic::skills::GetFootprintRequest& request,
      intrinsic::skills::GetFootprintContext& context) const override;

  absl::StatusOr<std::unique_ptr<google::protobuf::Message>> Preview(
      const intrinsic::skills::PreviewRequest& request,
      intrinsic::skills::PreviewContext& context) override;

  absl::StatusOr<std::unique_ptr<google::protobuf::Message>> Execute(
      const intrinsic::skills::ExecuteRequest& request,
      intrinsic::skills::ExecuteContext& context) override;

 private:
  // The zenoh session is created on first use and then reused: every trial
  // reads through this skill, and a per-invocation session would pay the
  // connect cost on the critical path.
  absl::StatusOr<intrinsic::KeyValueStore*> Store();

  absl::StatusOr<std::unique_ptr<google::protobuf::Message>> WritePoses(
      const ai::tar2::PoseKvStoreSkillParams& params);

  // Write path of a type that holds one pose under a bare key, such as home.
  absl::StatusOr<std::unique_ptr<google::protobuf::Message>> WriteSinglePose(
      const ai::tar2::PoseKvStoreSkillParams& params, const std::string& slug);

  absl::StatusOr<std::unique_ptr<google::protobuf::Message>> ReadPose(
      const ai::tar2::PoseKvStoreSkillParams& params);

  absl::Mutex mutex_;
  std::optional<intrinsic::PubSub> pubsub_ ABSL_GUARDED_BY(mutex_);
  std::optional<intrinsic::KeyValueStore> store_ ABSL_GUARDED_BY(mutex_);
};

#endif  // FLOWSTATE_AIC_KV_STORE_POSE_KV_STORE_SKILL_H_
