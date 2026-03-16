import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

# Define landmark indices
LEFT_SHOULDER = PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER.value
LEFT_ELBOW = PoseLandmark.LEFT_ELBOW.value
RIGHT_ELBOW = PoseLandmark.RIGHT_ELBOW.value
LEFT_WRIST = PoseLandmark.LEFT_WRIST.value
RIGHT_WRIST = PoseLandmark.RIGHT_WRIST.value
LEFT_HIP = PoseLandmark.LEFT_HIP.value
RIGHT_HIP = PoseLandmark.RIGHT_HIP.value
LEFT_KNEE = PoseLandmark.LEFT_KNEE.value
RIGHT_KNEE = PoseLandmark.RIGHT_KNEE.value
LEFT_ANKLE = PoseLandmark.LEFT_ANKLE.value
RIGHT_ANKLE = PoseLandmark.RIGHT_ANKLE.value
NOSE = PoseLandmark.NOSE.value
LEFT_EAR = PoseLandmark.LEFT_EAR.value
RIGHT_EAR = PoseLandmark.RIGHT_EAR.value

# Atomic start-to-end segments
LANDMARK_CHAINS = {
    "torso": [LEFT_SHOULDER, LEFT_HIP],             # shoulder → hip
    "left_upper_arm": [LEFT_SHOULDER, LEFT_ELBOW],  # shoulder → elbow
    "left_lower_arm": [LEFT_ELBOW, LEFT_WRIST],     # elbow → wrist
    "right_upper_arm": [RIGHT_SHOULDER, RIGHT_ELBOW],
    "right_lower_arm": [RIGHT_ELBOW, RIGHT_WRIST],
    "left_upper_leg": [LEFT_HIP, LEFT_KNEE],        # hip → knee
    "left_lower_leg": [LEFT_KNEE, LEFT_ANKLE],      # knee → ankle
    "right_upper_leg": [RIGHT_HIP, RIGHT_KNEE],
    "right_lower_leg": [RIGHT_KNEE, RIGHT_ANKLE],
    "neck_to_head": [NOSE, LEFT_EAR]               # nose → ear (or RIGHT_EAR)
}