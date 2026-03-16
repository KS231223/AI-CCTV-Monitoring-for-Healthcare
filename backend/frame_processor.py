import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
from landmark_chains import (
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from landmark_utils import to_numpy, filter_by_visibility
from entities import personFrame, Person
import math
 
 
def angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    a = p1 - p2
    b = p3 - p2
    dot = np.dot(a, b)
    mag = np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
    return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))
 
 
class DecoupledProcessor:
    """
    Stateless processor: converts a single frame + bbox (+ optional prev_center)
    into a personFrame entity. Uses the modern MediaPipe Tasks PoseLandmarker API.
 
    Args:
        model_path:           Path to the .task model bundle
                              (download: pose_landmarker_full.task from mediapipe releases)
        fps:                  Source video FPS, used for velocity calculation
        visibility_threshold: Minimum landmark visibility score to include
    """
 
    def __init__(
        self,
        model_path: str = "pose_landmarker_full.task",
        fps: int = 30,
        visibility_threshold: float = 0.7,
    ) -> None:
        self.fps = fps
        self.visibility_threshold = visibility_threshold
 
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
 
    def _fallback_frame(
        self,
        bbox: tuple[int, int, int, int],
        prev_center: np.ndarray | None,
    ) -> personFrame:
        """Return a dummy personFrame carrying only positional data."""
        x1, y1, x2, y2 = bbox
        bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        velocity = (
            0.0 if prev_center is None
            else float(np.linalg.norm(bbox_center - prev_center) * self.fps)
        )
        return personFrame(
            velocity_pixels_per_sec=velocity,
            bbox_center=bbox_center,
        )
 
    def process(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        prev_center: np.ndarray | None = None,
    ) -> personFrame:
        # Compute bbox_center and velocity up front — these are always available
        # from the bbox alone and are needed for both real and fallback frames.
        x1, y1, x2, y2 = bbox
        bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        velocity = (
            0.0 if prev_center is None
            else float(np.linalg.norm(bbox_center - prev_center) * self.fps)
        )
 
        try:
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                raise ValueError("Empty crop — bbox outside frame bounds")
 
            crop_h, crop_w = person_crop.shape[:2]
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
            result = self.landmarker.detect(mp_image)
 
            if not result.pose_landmarks:
                return personFrame(
                    velocity_pixels_per_sec=velocity,
                    bbox_center=bbox_center,
                )
 
            # Build landmark dict from the first (and only) detected pose
            raw = result.pose_landmarks[0]
            landmarks_dict = {
                i: {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for i, lm in enumerate(raw)
            }
            landmarks_dict = filter_by_visibility(
                landmarks_dict, threshold=self.visibility_threshold
            )
 
            if not landmarks_dict:
                return personFrame(
                    velocity_pixels_per_sec=velocity,
                    bbox_center=bbox_center,
                )
 
            coords = to_numpy(landmarks_dict, crop_w, crop_h)
 
            # Torso — guard individually so a missing hip/shoulder doesn't kill everything
            try:
                hip_mid = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2
                shoulder_mid = (coords[LEFT_SHOULDER] + coords[RIGHT_SHOULDER]) / 2
                torso = {
                    "x": float(hip_mid[0]),
                    "y": float(hip_mid[1]),
                    "height_pixels": float(np.linalg.norm(shoulder_mid - hip_mid)),
                }
            except KeyError:
                torso = None
 
            # Ankles / knees
            def _opt(idx: int) -> list | None:
                return coords[idx].tolist() if idx in coords else None
 
            left_ankle = _opt(LEFT_ANKLE)
            right_ankle = _opt(RIGHT_ANKLE)
            left_knee = _opt(LEFT_KNEE)
            right_knee = _opt(RIGHT_KNEE)
 
            # Knee angles
            def _knee_angle(hip_idx, knee_idx, ankle_idx) -> float | None:
                try:
                    if all(i in coords for i in (hip_idx, knee_idx, ankle_idx)):
                        return float(
                            angle_between(coords[hip_idx], coords[knee_idx], coords[ankle_idx])
                        )
                except (KeyError, ValueError):
                    pass
                return None
 
            left_knee_angle = _knee_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
            right_knee_angle = _knee_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
 
            # Shoulder sway
            try:
                sway = (
                    float(abs(coords[LEFT_SHOULDER][0] - coords[RIGHT_SHOULDER][0]))
                    if LEFT_SHOULDER in coords and RIGHT_SHOULDER in coords
                    else None
                )
            except KeyError:
                sway = None
 
            return personFrame(
                torso=torso,
                left_ankle=left_ankle,
                right_ankle=right_ankle,
                left_knee=left_knee,
                right_knee=right_knee,
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                sway=sway,
                bbox_center=bbox_center,
                velocity_pixels_per_sec=velocity,
            )
 
        except Exception:
            # Catch-all: any unexpected failure (corrupt frame, MediaPipe crash,
            # bad bbox, etc.) still yields a positional-only dummy so the buffer
            # stays continuous and downstream code never sees a missing entry.
            return personFrame(
                velocity_pixels_per_sec=velocity,
                bbox_center=bbox_center
            )
 
    def close(self) -> None:
        self.landmarker.close()
 
    def __enter__(self):
        return self
 
    def __exit__(self, *_):
        self.close()
 
 
class PersonAdapter:
    """
    Intermediary: converts DecoupledProcessor outputs into Person frame updates.
    Always appends to the buffer — even dummy frames — so the buffer length
    stays in sync with wall-clock frames and index-based analysis stays valid.
    """
 
    def __init__(self, processor: DecoupledProcessor, buffer_size: int = 120) -> None:
        self.processor = processor
        self.buffer_size = buffer_size
 
    def update_person(
        self,
        person: Person,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> personFrame:
        prev_center = (
            person.last_120_frames[-1].bbox_center if person.last_120_frames else None
        )
        frame_entity = self.processor.process(frame, bbox, prev_center=prev_center)
 
        # Always append — dummy frames preserve temporal continuity.
        person.last_120_frames.append(frame_entity)
        if len(person.last_120_frames) > self.buffer_size:
            person.last_120_frames.pop(0)
 
        return frame_entity
 
    