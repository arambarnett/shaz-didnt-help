from enum import Enum

class VideoStyle(Enum):
    FAST_ENERGIZED = "Fast & Energized"
    MODERATE = "Moderate"
    SLOW_SMOOTH = "Slow & Smooth"

# Define precise timing parameters for each style
STYLE_DEFINITIONS = {
    VideoStyle.FAST_ENERGIZED: {
        "cut_threshold": 0.100000,  # 100ms
        "transition_duration": 0.250000,  # 250ms
        "effect_timing": {
            "zoom": 0.500000,  # 500ms
            "speed": 0.750000,  # 750ms
            "text": 1.000000  # 1s
        },
        "speed_ranges": {
            "low": 1.500000,
            "medium": 2.000000,
            "high": 3.000000
        }
    },
    VideoStyle.MODERATE: {
        "cut_threshold": 0.250000,  # 250ms
        "transition_duration": 0.500000,  # 500ms
        "effect_timing": {
            "zoom": 0.750000,  # 750ms
            "speed": 1.000000,  # 1s
            "text": 1.500000  # 1.5s
        },
        "speed_ranges": {
            "low": 1.250000,
            "medium": 1.500000,
            "high": 2.000000
        }
    },
    VideoStyle.SLOW_SMOOTH: {
        "cut_threshold": 0.500000,  # 500ms
        "transition_duration": 1.000000,  # 1s
        "effect_timing": {
            "zoom": 1.000000,  # 1s
            "speed": 1.500000,  # 1.5s
            "text": 2.000000  # 2s
        },
        "speed_ranges": {
            "low": 1.100000,
            "medium": 1.250000,
            "high": 1.500000
        }
    }
}

def map_frontend_style_to_enum(style_name: str) -> VideoStyle:
    """Map frontend style names to VideoStyle enum with precise timing"""
    style_map = {
        "Fast & Energized": VideoStyle.FAST_ENERGIZED,
        "Moderate": VideoStyle.MODERATE,
        "Slow & Smooth": VideoStyle.SLOW_SMOOTH
    }
    return style_map.get(style_name, VideoStyle.MODERATE)

def get_style_timing_parameters(style: VideoStyle) -> dict:
    """Get precise timing parameters for a style"""
    return STYLE_DEFINITIONS[style]
