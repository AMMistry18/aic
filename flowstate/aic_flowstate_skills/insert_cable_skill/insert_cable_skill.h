#ifndef INSERT_CABLE_SKILL_H_
#define INSERT_CABLE_SKILL_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "intrinsic/skills/cc/skill_interface.h"
#include "intrinsic/skills/proto/skill_service.pb.h"

class InsertCableSkill final : public intrinsic::skills::SkillInterface {
 public:
  // ---------------------------------------------------------------------------
  // Skill signature
  // ---------------------------------------------------------------------------
  static std::unique_ptr<intrinsic::skills::SkillInterface> CreateSkill();

  // ---------------------------------------------------------------------------
  // Skill execution
  // ---------------------------------------------------------------------------
  absl::StatusOr<std::unique_ptr<google::protobuf::Message>> Execute(
      const intrinsic::skills::ExecuteRequest& request,
      intrinsic::skills::ExecuteContext& context) override;
};

#endif  // INSERT_CABLE_SKILL_H_
