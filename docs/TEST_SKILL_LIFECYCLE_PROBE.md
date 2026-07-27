# Test Skill v1 lifecycle probe

`ai.tar2.test_skill_v1` is a deliberately stateless temporary Flowstate skill.
It is a deployment/restart diagnostic, not part of the board-search policy.

It uses the same Dockerfile, SDK, image/bundle identity checks, `inbuild`
bundle mechanism, and gRPC port as the production v4 skill. It intentionally
does **not** import or use ROS, AIC model code, cameras, TF, controllers,
Move Robot, OpenCV, NumPy, or persistent state. Its only execution result is:

```text
success = true
message = "lifecycle probe ok"
```

## Build and install

From WSL:

```bash
cd /mnt/c/tmp/ws_aic_phase1
rsync -a --delete --exclude .git \
  /mnt/c/Users/anshu/College/aic/aic/ src/aic/
find src/aic/flowstate -type f \( -name '*.py' -o -name '*.sh' \) \
  -exec sed -i 's/\r$//' {} +
INBUILD_BIN=$PWD/inbuild \
  bash src/aic/flowstate/scripts/build_test_skill.sh

tar -tf images/test_skill_v1/test_skill_v1.bundle.tar |
  grep -x test_skill_v1.tar

AIC_SOLUTION=162d7a70-b696-4260-974d-fdae049e6eaa_BRANCH \
  bash install_skill.sh images/test_skill_v1/test_skill_v1.bundle.tar
```

Wait for Flowstate's asset-upload notice to finish before stopping the
solution.

## Experiment

1. Add **Test Skill v1 (Lifecycle Probe)** to a minimal saved Flowstate
   process. It has no parameters and requires no controller/lifecycle node.
2. Run it while the solution is already running. It must return
   `success=true` and `lifecycle probe ok`.
3. Stop the solution normally, wait for it to reach fully stopped, then start
   it again.
4. Run the unchanged saved process again.

Interpretation:

| Result | Meaning |
| --- | --- |
| Probe works before and after restart | v4-specific packaging or runtime remains suspect. |
| Probe works before restart but fails with Executive 18100 after restart | Generic Flowstate custom-skill asset/workload reconciliation issue; neither v4 search nor the AIC model caused it. |
| Probe fails before the first restart | Generic custom-skill install/start issue; inspect the probe's skill logs. |
