"""Save one ground-truth/prediction rollout without opening a viewer."""

from pathlib import Path

import rerun as rr


def save_rollout(path, predictions, ground_truth):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream("3dpwm")
    recording.save(str(path))
    with recording:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for step, (prediction, target) in enumerate(zip(predictions, ground_truth)):
            rr.set_time_sequence("step", step)
            rr.log(
                "world/ground_truth",
                rr.Points3D(target, colors=[70, 170, 255], radii=0.002),
            )
            rr.log(
                "world/prediction",
                rr.Points3D(prediction, colors=[255, 140, 60], radii=0.002),
            )
    recording.flush()
