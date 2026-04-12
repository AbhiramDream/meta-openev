import math

def calculate_safe_score(raw_value: float, max_value: float) -> float:
    """
    Normalises a raw score to the (0.01, 0.99) range strictly.
    Uses a sigmoid-like approach to ensure it never hits 0 or 1.
    """
    if max_value <= 0:
        max_value = 1.0
        
    # Standard normalisation [0, 1]
    norm = max(0.0, min(1.0, raw_value / max_value))
    
    # Map [0, 1] to [0.01, 0.99]
    return 0.01 + (0.98 * norm)
