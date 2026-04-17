def calculate_weighted_rating(v: float, R: float, m: float, C: float) -> float:
    """Tính điểm phổ biến dựa trên công thức của IMDB"""
    if v + m == 0:
        return 0
    return (v / (v + m) * R) + (m / (v + m) * C)