# AIC model runtime

`aic_model/aic_model/` contains the ROS lifecycle host and the current cable
insertion policy.

| Path | Purpose |
| --- | --- |
| `aic_model.py` | ROS lifecycle node that loads a configured policy |
| `policy.py` | Base policy interface supplied by the AIC framework |
| `insertion/InsertionPolicy.py` | Active SFP and SC insertion entry point |
| `insertion/sfp_controller.py` | Current SFP insertion controller |
| `insertion/sc_controller.py` | Current SC insertion controller |
| `insertion/*_pose*.py` | Plug/port perception and geometry |
| `insertion/board_search.py` | Bounded task-board framing |
| `insertion/visual_gap*.py` | Visual recovery helpers |

The configured ROS policy name is:

```text
aic_model.insertion.InsertionPolicy
```

Docker builds copy this package directly. There is no separate policy overlay.
The legacy `RL_INSERT_V50_*` environment-variable names remain accepted so
existing deployment configuration does not break; they configure
`sfp_controller.py` and do not identify a separate source version.
