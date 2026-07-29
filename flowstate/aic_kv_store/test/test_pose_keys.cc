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

#include <gtest/gtest.h>

#include "pose_keys.h"

namespace {

using Params = ::ai::tar2::PoseKvStoreSkillParams;

::intrinsic_proto::Pose UnitPose() {
  ::intrinsic_proto::Pose pose;
  pose.mutable_position()->set_x(0.1);
  pose.mutable_position()->set_y(0.2);
  pose.mutable_position()->set_z(0.3);
  pose.mutable_orientation()->set_w(1.0);
  return pose;
}

TEST(ParseTargetName, ReadsTheThreeProcessInputNames) {
  const auto sfp = aic_kv::ParseTargetName("sfp_mount_2");
  ASSERT_TRUE(sfp.ok()) << sfp.status();
  EXPECT_EQ(sfp->type, Params::OBJECT_TYPE_SFP);
  EXPECT_EQ(sfp->index, 2);

  const auto nic = aic_kv::ParseTargetName("nic_card_mount_3");
  ASSERT_TRUE(nic.ok()) << nic.status();
  EXPECT_EQ(nic->type, Params::OBJECT_TYPE_NIC);
  EXPECT_EQ(nic->index, 3);

  const auto sc = aic_kv::ParseTargetName("sc_port_1");
  ASSERT_TRUE(sc.ok()) << sc.status();
  EXPECT_EQ(sc->type, Params::OBJECT_TYPE_SC);
  EXPECT_EQ(sc->index, 1);
}

TEST(ParseTargetName, IgnoresDecorationAroundTheName) {
  for (const char* name : {"  NIC_Card_Mount_4 ", "nic-card-mount-4",
                           "task_board/nic_card_mount_4"}) {
    const auto target = aic_kv::ParseTargetName(name);
    ASSERT_TRUE(target.ok()) << name << ": " << target.status();
    EXPECT_EQ(target->type, Params::OBJECT_TYPE_NIC) << name;
    EXPECT_EQ(target->index, 4) << name;
  }
}

TEST(ParseTargetName, AcceptsARailInfix) {
  const auto target = aic_kv::ParseTargetName("sfp_mount_rail_0");
  ASSERT_TRUE(target.ok()) << target.status();
  EXPECT_EQ(target->type, Params::OBJECT_TYPE_SFP);
  EXPECT_EQ(target->index, 0);
}

TEST(ParseTargetName, ReadsHomeAsAnIndexLessType) {
  for (const char* name : {"home", " HOME ", "aic/phase1/home", "home_pose"}) {
    const auto target = aic_kv::ParseTargetName(name);
    ASSERT_TRUE(target.ok()) << name << ": " << target.status();
    EXPECT_EQ(target->type, Params::OBJECT_TYPE_HOME) << name;
    EXPECT_EQ(target->index, aic_kv::kNoIndex) << name;
  }
}

TEST(TypeHasIndex, IsFalseOnlyForHome) {
  EXPECT_TRUE(aic_kv::TypeHasIndex(Params::OBJECT_TYPE_SFP));
  EXPECT_TRUE(aic_kv::TypeHasIndex(Params::OBJECT_TYPE_NIC));
  EXPECT_TRUE(aic_kv::TypeHasIndex(Params::OBJECT_TYPE_SC));
  EXPECT_FALSE(aic_kv::TypeHasIndex(Params::OBJECT_TYPE_HOME));
}

TEST(ParseTargetName, RejectsNamesThatCarryNoIndexOrNoType) {
  EXPECT_FALSE(aic_kv::ParseTargetName("").ok());
  EXPECT_FALSE(aic_kv::ParseTargetName("nic_card_mount").ok());
  EXPECT_FALSE(aic_kv::ParseTargetName("mystery_mount_2").ok());
}

TEST(ParseTargetName, RejectsAnIndexOutsideTheFiveLabels) {
  EXPECT_FALSE(aic_kv::ParseTargetName("sc_port_5").ok());
  EXPECT_FALSE(aic_kv::ParseTargetName("nic_card_mount_12").ok());
}

TEST(MakePoseKey, UsesTheDefaultPrefixAndTheLabelAsTheLastSegment) {
  const auto key =
      aic_kv::MakePoseKey("", Params::OBJECT_TYPE_SFP, 3);
  ASSERT_TRUE(key.ok()) << key.status();
  EXPECT_EQ(*key, "aic/phase1/sfp/3");
}

TEST(MakePoseKey, NormalizesAnExplicitPrefix) {
  const auto key =
      aic_kv::MakePoseKey("/aic/phase1_test/", Params::OBJECT_TYPE_SC, 0);
  ASSERT_TRUE(key.ok()) << key.status();
  EXPECT_EQ(*key, "aic/phase1_test/sc/0");
}

TEST(MakePoseKey, GivesHomeABareKeyAndIgnoresTheIndex) {
  const auto key =
      aic_kv::MakePoseKey("", Params::OBJECT_TYPE_HOME, aic_kv::kNoIndex);
  ASSERT_TRUE(key.ok()) << key.status();
  EXPECT_EQ(*key, "aic/phase1/home");

  const auto stray_index =
      aic_kv::MakePoseKey("", Params::OBJECT_TYPE_HOME, 3);
  ASSERT_TRUE(stray_index.ok()) << stray_index.status();
  EXPECT_EQ(*stray_index, "aic/phase1/home");

  const auto prefixed =
      aic_kv::MakePoseKey("aic/phase1_test", Params::OBJECT_TYPE_HOME,
                          aic_kv::kNoIndex);
  ASSERT_TRUE(prefixed.ok()) << prefixed.status();
  EXPECT_EQ(*prefixed, "aic/phase1_test/home");
}

TEST(MakePoseKey, RejectsUnusableIndicesTypesAndPrefixes) {
  EXPECT_FALSE(aic_kv::MakePoseKey("", Params::OBJECT_TYPE_NIC, 5).ok());
  EXPECT_FALSE(aic_kv::MakePoseKey("", Params::OBJECT_TYPE_NIC, -1).ok());
  EXPECT_FALSE(
      aic_kv::MakePoseKey("", Params::OBJECT_TYPE_UNSPECIFIED, 0).ok());
  EXPECT_FALSE(aic_kv::MakePoseKey("aic/*", Params::OBJECT_TYPE_NIC, 0).ok());
}

TEST(ValidatePose, AcceptsAUsablePose) {
  EXPECT_TRUE(aic_kv::ValidatePose(UnitPose(), "pose").ok());
}

TEST(ValidatePose, RejectsAnUnsetOrZeroQuaternion) {
  ::intrinsic_proto::Pose no_orientation;
  no_orientation.mutable_position()->set_x(0.1);
  EXPECT_FALSE(aic_kv::ValidatePose(no_orientation, "pose").ok());

  ::intrinsic_proto::Pose zero_quaternion = UnitPose();
  zero_quaternion.mutable_orientation()->set_w(0.0);
  EXPECT_FALSE(aic_kv::ValidatePose(zero_quaternion, "pose").ok());
}

TEST(PoseIsUiDefault, TreatsAllZerosAndIdentityAtOriginAsFlowstateDefault) {
  ::intrinsic_proto::Pose empty;
  EXPECT_TRUE(aic_kv::PoseIsUiDefault(empty));

  ::intrinsic_proto::Pose zeros;
  zeros.mutable_position();
  zeros.mutable_orientation();
  EXPECT_TRUE(aic_kv::PoseIsUiDefault(zeros));

  ::intrinsic_proto::Pose identity_at_origin;
  identity_at_origin.mutable_position();
  identity_at_origin.mutable_orientation()->set_w(1.0);
  EXPECT_TRUE(aic_kv::PoseIsUiDefault(identity_at_origin));

  EXPECT_FALSE(aic_kv::PoseIsUiDefault(UnitPose()));
}

}  // namespace
