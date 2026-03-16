from typing import List, Dict

class Person:
    def __init__(
        self,
        bbox: List[int] = None,
        person_id: str = "Unknown",
        height: int = 185,
        weights: Dict = None,
        identified: bool = False,
        last_120_frames: List = None,
        last_120_rPPG : List = None,
        heart_rate : float = 0,
        rppg_variance : float = 0
    ):
        self.person_id: str = person_id
        self.display_id: str = person_id[:7]  # first 5 chars for display
        self.height: int = height
        self.weights: Dict = weights if weights else {}
        self.identified: bool = identified
        self.last_120_frames: List = last_120_frames if last_120_frames else []
        self.last_120_rPPG : List = last_120_rPPG if last_120_frames else []
        self.heart_rate : float = heart_rate
        self.rppg_variance : float = rppg_variance
        self.bbox: List[int] = bbox if bbox else [0,0,0,0]

    def __repr__(self):
        return (f"<Person {self.display_id} | Identified: {self.identified} | "
                f"Height: {self.height} | Frames stored: {len(self.last_120_frames)}>")
    

class personFrame:
    """Pure data entity representing gait info for a single frame"""
    def __init__(
        self,
        torso=None,
        left_ankle=None,
        right_ankle=None,
        left_knee=None,
        right_knee=None,
        left_knee_angle=None,
        right_knee_angle=None,
        sway=None,
        bbox_center=None,
        velocity_pixels_per_sec=None
    ):
        self.torso = torso
        self.left_ankle = left_ankle
        self.right_ankle = right_ankle
        self.left_knee = left_knee
        self.right_knee = right_knee
        self.left_knee_angle = left_knee_angle
        self.right_knee_angle = right_knee_angle
        self.sway = sway
        self.bbox_center = bbox_center
        self.velocity_pixels_per_sec = velocity_pixels_per_sec

    def to_dict(self):
        return self.__dict__