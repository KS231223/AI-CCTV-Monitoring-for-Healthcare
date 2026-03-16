# body_segments.py
from landmark_chains import LANDMARK_CHAINS

# Unified dictionary combining metrics with references to the atomic chains
BODY_METRICS = {
    # Torso
    "torso": {
        "landmarks": LANDMARK_CHAINS["torso"],
        "proportional_length": 0.50,
        "uncertainty": 0.05
    },
    # Left Arm
    "left_upper_arm": {
        "landmarks": LANDMARK_CHAINS["left_upper_arm"],
        "proportional_length": 0.25,
        "uncertainty": 0.05
    },
    "left_lower_arm": {
        "landmarks": LANDMARK_CHAINS["left_lower_arm"],
        "proportional_length": 0.25,
        "uncertainty": 0.05
    },
    # Right Arm
    "right_upper_arm": {
        "landmarks": LANDMARK_CHAINS["right_upper_arm"],
        "proportional_length": 0.25,
        "uncertainty": 0.05
    },
    "right_lower_arm": {
        "landmarks": LANDMARK_CHAINS["right_lower_arm"],
        "proportional_length": 0.25,
        "uncertainty": 0.05
    },
    # Left Leg
    "left_upper_leg": {
        "landmarks": LANDMARK_CHAINS["left_upper_leg"],
        "proportional_length": 0.45,
        "uncertainty": 0.05
    },
    "left_lower_leg": {
        "landmarks": LANDMARK_CHAINS["left_lower_leg"],
        "proportional_length": 0.45,
        "uncertainty": 0.05
    },
    # Right Leg
    "right_upper_leg": {
        "landmarks": LANDMARK_CHAINS["right_upper_leg"],
        "proportional_length": 0.45,
        "uncertainty": 0.05
    },
    "right_lower_leg": {
        "landmarks": LANDMARK_CHAINS["right_lower_leg"],
        "proportional_length": 0.45,
        "uncertainty": 0.05
    },
    # Neck / Head
    "neck_to_head": {
        "landmarks": LANDMARK_CHAINS["neck_to_head"],
        "proportional_length": 0.15,
        "uncertainty": 0.03
    },
    # Total height
    "height": {
        "value": 1.85,
        "uncertainty": 0.0
    }
}