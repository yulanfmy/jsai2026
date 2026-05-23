"""Russell's circumplex emotion model with 11 emotion states.

Each emotion is mapped to (arousal, valence) coordinates in [-1, 1] space,
along with recommended BPM range.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Emotion:
    name: str
    arousal: float
    valence: float
    bpm_low: int
    bpm_high: int
    description: str


EMOTIONS: dict[str, Emotion] = {
    "angry": Emotion("Angry", 0.8, -0.7, 130, 170, "High energy, negative mood"),
    "anxious": Emotion("Anxious", 0.7, -0.5, 110, 150, "Restless and uneasy"),
    "sad": Emotion("Sad", -0.5, -0.7, 50, 80, "Low energy, deep sorrow"),
    "melancholy": Emotion("Melancholy", -0.3, -0.4, 60, 90, "Gentle sadness, reflective"),
    "tired": Emotion("Tired", -0.8, -0.2, 50, 70, "Very low energy, fatigued"),
    "restless": Emotion("Restless", 0.4, -0.2, 100, 130, "Unsettled, seeking change"),
    "calm": Emotion("Calm", -0.5, 0.4, 60, 85, "Relaxed and at ease"),
    "peaceful": Emotion("Peaceful", -0.6, 0.6, 55, 80, "Serene and content"),
    "focused": Emotion("Focused", 0.2, 0.3, 90, 120, "Alert and concentrated"),
    "confident": Emotion("Confident", 0.5, 0.7, 100, 140, "Assured and empowered"),
    "excited": Emotion("Excited", 0.9, 0.8, 120, 160, "High energy, very positive"),
}


def get_emotion(name: str) -> Emotion:
    key = name.lower().strip()
    if key not in EMOTIONS:
        raise ValueError(f"Unknown emotion: {name}. Choose from: {list(EMOTIONS.keys())}")
    return EMOTIONS[key]


def list_emotions() -> list[Emotion]:
    return list(EMOTIONS.values())
