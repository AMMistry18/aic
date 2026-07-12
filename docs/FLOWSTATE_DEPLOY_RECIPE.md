# Flowstate aic_model deploy recipe (inctl / inbuild)

Recovered 2026-07-12. The `inctl` + `inbuild` binaries and all bundles live only
in `/private/tmp`, which is wiped on reboot. This file is the durable record of
the exact deploy procedure so it never has to be re-derived from chat history.

## Tooling (both were at /private/tmp, both wiped by reboot)

```text
inctl:   /private/tmp/inctl-linux-amd64          (linux/amd64, ~39 MB)
inbuild: /private/tmp/aic-flowstate-guided-v5/inbuild   (linux/amd64)
inctl home (auth/session): /private/tmp/aic-inctl-home
macOS CA file: /etc/ssl/cert.pem
```

Both `inctl` and `inbuild` are downloaded from the Flowstate console
(flowstate.intrinsic.ai, "set up development environment" / dev tools). They are
not in the repo and there is no public GitHub release. To survive future reboots,
copy them somewhere persistent (e.g. `~/.aic-inctl/`) after downloading.

All CLI calls run the linux binary inside a debian container on this Mac:

```bash
docker run --rm --platform linux/amd64 \
  -v /etc/ssl/cert.pem:/etc/ssl/certs/ca-certificates.crt:ro \
  -v /private/tmp/inctl-linux-amd64:/inctl:ro \
  -v /private/tmp/aic-inctl-home:/root \
  debian:bookworm-slim \
  /inctl <command>
```

## Auth

```bash
/inctl auth login --no_browser --org tar-2@xfa-prod-aic-us
```

Prints a URL (flowstate.intrinsic.ai/o/tar-2@xfa-prod-aic-us/generate-keys). User
opens it, approves, and pastes the token INTO THE CLI PROMPT (needs an interactive
TTY: `docker run -i`). Never paste the token into chat, scripts, or Git.

## Target

```text
org:      tar-2@xfa-prod-aic-us
solution: 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH   (Copy of AIC Phase 1 Template)
cluster:  verify with `inctl cluster list` (was vmp-f5ed-053nou72)
service instance name: aic_model   (KEEP THIS — it is the ROS lifecycle node name)
ROS lifecycle node:    aic_model   -> /aic_model/change_state
```

## Build the image (thin overlay — fast, only flips one env)

The base `my-solution:student-flowstate-guided-v5` (sha b2a56d8387df) already has
the correct code/models/entrypoint. To change only the control mode:

```bash
cat > /tmp/Dockerfile.overlay <<'EOF'
FROM my-solution:student-flowstate-guided-v5
ENV RL_INSERT_CONTROL_MODE=rl     # or guided
EOF
docker build --platform linux/amd64 -f /tmp/Dockerfile.overlay \
  -t my-solution:student-flowstate-rl-v8 .
```

## Bump the asset version (REQUIRED — Flowstate keys assets by manifest identity)

Reinstalling the SAME asset id/version does NOT replace the running image. You must
change the manifest `name` each time: aic_model_v6 -> v7 -> v8 -> ...
Asset id becomes `ai.intrinsic.<name>`. The version suffix (`0.0.1+<hash>`) is
content-derived; the `name` is what makes it a new identity.

Manifest is a textproto; only these fields change per version:

```textproto
metadata {
  id { package: "ai.intrinsic"  name: "aic_model_v8" }   # <- bump this
  vendor { display_name: "Intrinsic" }
  documentation { description: "Participant Policy Node (rl mode v8)." }
  display_name: "Participant Policy Node v8"
}
service_def {
  real_spec { image { archive_filename: "aic_model.tar"
    settings { resource_requirements { limits { key: "nvidia.com/gpu" value: "1" } } } } }
  sim_spec  { image { archive_filename: "aic_model.tar"
    settings { resource_requirements { limits { key: "nvidia.com/gpu" value: "1" } } } } }
}
assets { image_filenames: ["aic_model.tar"] }
```

## Build the bundle

Two ways; either works. The bundle is just `aic_model.tar` (docker save) +
`service_manifest.binarypb` at the tar root.

**A. With inbuild (preferred — it compiles the textproto to binarypb):**

```bash
WORK=/private/tmp/aic-flowstate-v8
mkdir -p $WORK/images/aic_model
docker save my-solution:student-flowstate-rl-v8 \
  -o $WORK/images/aic_model/aic_model.tar          # ~7 GB, minutes
# put aic_model_v8.manifest.textproto in $WORK
docker run --rm --platform linux/amd64 \
  -v /private/tmp/aic-flowstate-guided-v5/inbuild:/inbuild:ro \
  -v $WORK:/work -v $WORK/images/aic_model:/img:ro \
  debian:bookworm-slim \
  /inbuild service bundle \
    --manifest /work/aic_model_v8.manifest.textproto \
    --oci_image /img/aic_model.tar \
    --output /work/aic_model_v8.bundle.tar
```

**B. Without inbuild (hand-pack; reuse a known-good binarypb):**
Extract `service_manifest.binarypb` from a prior bundle, swap the identity strings
(package/name/description/display_name are all <128 bytes so each is a single-byte
length-delimited token; adjust that byte AND every enclosing message length), then:

```bash
cp $WORK/images/aic_model/aic_model.tar ./aic_model.tar
COPYFILE_DISABLE=1 tar --no-mac-metadata -cf aic_model_v8.bundle.tar \
  aic_model.tar service_manifest.binarypb    # exactly 2 entries, NO ._ AppleDouble
rm ./aic_model.tar
```

GOTCHA: macOS `tar` adds `._*` AppleDouble files that break the upload. Always use
`COPYFILE_DISABLE=1 tar --no-mac-metadata` and verify `tar -tf` shows exactly the
two members with no `._` entries.

## Install the asset (uploads ~7.5 GB — slow; upload may need a retry)

Stage into a docker volume to avoid a huge bind-mount, then:

```bash
SOL=582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
/inctl asset install /work/aic_model_v8.bundle.tar \
  --org tar-2@xfa-prod-aic-us --solution $SOL
```

If it stalls or the UI says "hasn't finished uploading yet", the session likely
expired (401/XSRF). Re-auth and re-run the SAME install; it is idempotent.

## Rebind the service instance to the new asset

```bash
/inctl service delete aic_model --org tar-2@xfa-prod-aic-us --solution $SOL
/inctl service add ai.intrinsic.aic_model_v8 \
  --name aic_model --org tar-2@xfa-prod-aic-us --solution $SOL
```

Keep `--name aic_model` so the ROS node stays `aic_model` and
`/aic_model/change_state` resolves.

## Verify

```bash
/inctl service state list --org ... --solution $SOL   # aic_model must appear
/inctl logs --org ... --solution $SOL --service aic_model --since 10m --tail 200
```

Then in the Flowstate UI: lifecycle skill node name = `aic_model`; configure ->
activate; run one insertion. Confirm the log prints `control=rl` (or `guided`).
`resource not found` for `--service aic_model` means the instance is not bound.

## GOTCHA: crash-loop on "AIC_MODEL_ROUTER_ADDR must be provided"

Symptom: after `service add`, the instance stays `NotFound` forever; `inctl logs
--service aic_model` shows only `AIC_MODEL_ROUTER_ADDR must be provided` and no
`Loading policy module` / `on_configure`. This is a crash loop, NOT slow
scheduling — polling will never succeed.

Cause: the image was built as a thin overlay `FROM ...guided-v5`, which still
carries the OLD strict `/entrypoint.sh` that does `exit 1` when no router env is
injected. Overlaying only `ENV RL_INSERT_CONTROL_MODE=rl` does NOT replace that
baked entrypoint. Flowstate does not always inject `AIC_MODEL_ROUTER_ADDR`, so
the strict entrypoint exits immediately.

Fix: build from the FULL `docker/aic_model/Dockerfile.student_flowstate` (which
bakes `ENV AIC_ROUTER_ADDR=zenoh-router.app-intrinsic-base.svc.cluster.local:7447`
and installs the non-fatal entrypoint that falls back to peer-scouting), NOT an
overlay on guided-v5. Verify the built image has BOTH baked before bundling:

```bash
docker inspect <img> --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'RL_INSERT_CONTROL_MODE|AIC_ROUTER_ADDR'   # both must be present
# and the baked /entrypoint.sh must NOT `exit 1` on missing router (only under AIC_ENABLE_ACL)
```

## Cleanup

After v8 is bound and working, uninstall stale assets so only the current one
remains: old `ai.intrinsic.aic_model`, `aic_model_v6/v7`, and
`ai.tar2.aic_insertion_policy`. Do NOT delete the solution or unrelated services.
