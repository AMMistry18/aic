# Troubleshooting

## Low real-time factor on Gazebo

The simulation is configured to run at **1.0 RTF (100% real-time factor)**, meaning simulation time should match wall-clock time. If you're experiencing lower RTF, the following sections may help diagnose and resolve the issue.

### Gazebo not using the dedicated GPU

If your machine has two GPUs (or a CPU with an integrated GPU), OpenGL may be using the *integrated* GPU for rendering, which causes RTF to be very low. To fix this, you may need to manually force it to use the *discrete* GPU.

To check if Open GL is using the discrete GPU, run `glxinfo -B`. The output should show the details of your discrete GPU. Additionally, you can verify GPU-specific process by running `nvidia-smi`. When the AIC sim is active, `gz sim` should appear in the process list.

If the wrong GPU is selected, run `sudo prime-select nvidia`.
**Note**: You must log out and log in again for the changes to take effect. Then, re-run `glxinfo -B` to verify that the discrete GPU is active.

You can also check out [Problems with dual Intel and Nvidia GPU systems](https://gazebosim.org/docs/latest/troubleshooting/#problems-with-dual-intel-and-nvidia-gpu-systems).

### No GPU Available

If your system doesn't have a dedicated GPU, you may experience poor real-time factor (RTF) performance. This is because Gazebo uses [GlobalIllumination (GI)](https://gazebosim.org/api/sim/9/global_illumination.html) based rendering for the AIC scene, which requires GPU acceleration for optimal performance.

**To improve simulation performance on systems without a GPU:**

You can disable GlobalIllumination by editing [`aic.sdf`](../aic_description/world/aic.sdf) and setting `<enabled>` to `false` in the global illumination configuration [here](https://github.com/intrinsic-dev/aic/blob/c8aa4571d9dc4bd55bbefc02b0a160ba0e8e1e90/aic_description/world/aic.sdf#L39) and [here](https://github.com/intrinsic-dev/aic/blob/c8aa4571d9dc4bd55bbefc02b0a160ba0e8e1e90/aic_description/world/aic.sdf#L109). This will reduce rendering quality but may significantly improve RTF on CPU-only systems.

> [!WARNING]
> Disabling GI will change the visual appearance of the scene, which may affect vision-based policies.

## Zenoh Shared Memory Watchdog Warnings

When running the system, you may see warnings like:

```
WARN Watchdog Validator ThreadId(17) zenoh_shm::watchdog::periodic_task:
error setting scheduling priority for thread: OS(1), will run with priority 48.
This is not an hard error and it can be safely ignored under normal operating conditions.
```

**This warning is harmless and can be safely ignored.** It indicates that Zenoh's shared memory watchdog thread couldn't set a higher scheduling priority (which requires elevated privileges). The system will continue to work correctly.

**Why it happens:**
- The watchdog thread monitors shared memory health
- Setting higher priority requires `CAP_SYS_NICE` capability or root privileges
- Without it, the thread runs at default priority (48)

**When it might matter:**
- Under extremely high CPU load, the watchdog may occasionally miss its deadlines
- This could cause rare timeouts in shared memory operations
- In practice, this is almost never an issue for typical workloads

**To verify shared memory is working:**
```bash
# Check for Zenoh shared memory files
ls -lh /dev/shm | grep zenoh

# Monitor network traffic (should be minimal)
sudo tcpdump -i lo port 7447 -v
```

If you see Zenoh files in `/dev/shm` and minimal traffic on port 7447, shared memory is functioning correctly despite the warning.

## NVIDIA RTX 50xx cards not supported by PyTorch version locked in Pixi

```
UserWarning:
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.

The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

The `lerobot` version in `pixi.toml` depends on an older version of `pytorch` (built for an older version of cuda). 
`pixi install` will pull in that older version which does not support the newer sm_120 architecture for NVIDIA RTX 50xx cards.

We were able to run this policy on an Nvidia RTX 5090 by adding the following to `pixi.toml`:
```
[pypi-options.dependency-overrides]
torch = ">=2.7.1"
torchvision = ">=0.22.1"
```

See this [LeRobot issue](https://github.com/huggingface/lerobot/issues/2217) for details.

## Error: no such container aic_eval

when running `distrobox enter -r aic_eval`, you might encounter the following error:
```bash
Error: no such container aic_eval
```

By default, distrobox uses podman but we are using docker in our setup. Make sure to have set the default container manager by exporting the `DBX_CONTAINER_MANAGER` environment variable:
```bash
export DBX_CONTAINER_MANAGER=docker
```

## Distrobox: "First time user password setup" / `passwd` errors

**What it is:** With **`distrobox create -r`** (rootful), Distrobox marks your user for a **one-time interactive `passwd`** on the first login shell. In a **non-interactive** session (automation, some IDE terminals, `distrobox enter ... -- command`), `passwd` cannot run and you may see:

```text
⚠️  First time user password setup ⚠️
passwd: Authentication token manipulation error
```

or the shell appears to hang on `Current password:`.

**Fix A — Create without the prompt (recommended for new boxes):** Stop/remove the old box if needed, then recreate using the same image and Distrobox’s documented skip flag (see [Getting Started — Step 2](./getting_started.md#step-2-start-the-evaluation-container)):

```bash
export DBX_CONTAINER_MANAGER=docker
distrobox rm -f aic_eval   # only if you are OK removing the existing box
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval \
  --absolutely-disable-root-password-i-am-really-positively-sure
```

This mounts Distrobox’s `/run/.nopasswd` marker so init does **not** install the first-login `passwd` hook. It weakens the **container user** password policy; treat the box as a local sim environment only.

**Fix B — One interactive login:** From a normal terminal (TTY), run `distrobox enter -r aic_eval`, set a password when `passwd` prompts, then exit. Later non-interactive `distrobox enter` runs will skip the prompt.

**Fix C — Existing container, host has Docker:** Remove the first-login stamp as **root** in the container (username is usually the same as on the host, often `ubuntu`):

```bash
CID="$(docker ps -qf name=aic_eval)"
docker exec -u 0 "$CID" rm -f "/var/tmp/.${USER}.passwd.initialize"
```

Then `distrobox enter -r aic_eval` (or `distrobox enter -r aic_eval -- your-command`) should proceed without the password banner.

**Fix D — Sudo asking for a password** when Distrobox runs `docker`/`podman` as root: configure your sudo program to allow those commands without a password (narrow `NOPASSWD` rule), as described in the [Distrobox FAQ](https://github.com/89luca89/distrobox/blob/main/docs/posts/run_rootless.md) and related issues.

## TF warnings: `Detected jump back in time` / `Moved backwards in time`

You may see bursts of messages such as:

```text
[tf2_buffer]: Detected jump back in time. Clearing TF buffer.
[robot_state_publisher]: Moved backwards in time, re-publishing joint transforms!
```

**What it means:** `tf2` tracks **ROS time** from `/clock` (simulation time when `use_sim_time` is true). If that time **decreases**, buffers are cleared so transforms stay consistent. That is **normal** when:

- Gazebo is still starting or loading the world
- The **aic_engine** resets or readies the simulator for a **trial** (world / clock can jump)
- The machine is **CPU- or GPU-bound** and the sim **lags** (real-time factor &lt; 1), so `/clock` can look uneven

**When to worry:** Short bursts at startup or at trial boundaries are expected. **Continuous** warnings with no progress (spawners stuck, RTF very low) usually mean overload or a bad state—try closing other heavy jobs, improving RTF (see [Low real-time factor](#low-real-time-factor-on-gazebo)), or restarting `/entrypoint.sh` once.

**What not to do:** Do not run **two** `aic_model` nodes or mix middleware configs; duplicate nodes also confuse discovery and can coincide with odd timing.

There is no separate product bug in `aic_adapter` for these lines—they are `tf2` reacting to **clock non-monotonicity** from the simulator clock.

## Policy runs locally but fails silently on the portal

If a policy passes the local verification test but fails on the portal with no error or stdout logs, the cause is often one of the following:

### `task.time_limit` is enforced on the simulation clock

The task time limit is measured against simulation time (the ROS clock), not wall-clock time. A policy that relies on `time.time()` or `time.sleep()` may work locally (where sim time closely tracks wall time at ~1.0 RTF) but misbehave on the portal if sim and wall time diverge. Use the ROS clock for any time-based logic inside the policy.

### Heavy top-level imports count against the 30-second discovery budget

When the policy module is loaded, all top-level code, including imports, runs within a 30-second model discovery budget (`model_discovery_timeout_seconds`). Importing large libraries such as `torch` at the top of the module can exceed this budget and cause the policy to be killed before it reports an error.

Move heavy imports into the policy class `__init__` instead. The policy class is instantiated inside `on_configure`, which has its own 60-second budget (`model_configure_timeout_seconds`), giving imports more headroom and keeping discovery fast.
