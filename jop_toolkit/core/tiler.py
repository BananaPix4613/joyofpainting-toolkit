"""Joy of Painting canvas layout: auto (single size) and mixed greedy packing."""

CANVAS_SIZES = [(16, 16), (16, 32), (32, 16), (32, 32)]


def choose_auto_tile_size(width, height):
    """Pick the single canvas size that uses the fewest canvases."""
    best, best_score = None, None
    for tw, th in CANVAS_SIZES:
        cols = (width + tw - 1) // tw
        rows = (height + th - 1) // th
        canvas_count = cols * rows
        padding = cols * tw * rows * th - width * height
        score = (canvas_count, padding)
        if best_score is None or score < best_score:
            best_score, best = score, (tw, th)
    return best


def greedy_mixed_layout(width, height):
    """Pack width x height (multiples of 16) using mixed canvas sizes, largest first."""
    if width % 16 or height % 16:
        raise ValueError("Mixed layout requires width and height to be multiples of 16")
    cols, rows = width // 16, height // 16
    covered = [[False] * cols for _ in range(rows)]
    sizes = sorted(CANVAS_SIZES, key=lambda s: -s[0] * s[1])

    placements = []
    for r in range(rows):
        for c in range(cols):
            if covered[r][c]:
                continue
            for tw, th in sizes:
                tcw, tch = tw // 16, th // 16
                if r + tch > rows or c + tcw > cols:
                    continue
                if all(not covered[r + dr][c + dc]
                       for dr in range(tch) for dc in range(tcw)):
                    placements.append({'pixel_pos': [c * 16, r * 16], 'tile_size': [tw, th]})
                    for dr in range(tch):
                        for dc in range(tcw):
                            covered[r + dr][c + dc] = True
                    break
    return placements


def uniform_layout(width, height, tile_size):
    """Tile width x height with a single canvas size. Image must be a multiple of the tile size."""
    tw, th = tile_size
    if width % tw or height % th:
        raise ValueError("Image must be a multiple of the chosen canvas size")
    placements = []
    for ry in range(0, height, th):
        for cx in range(0, width, tw):
            placements.append({'pixel_pos': [cx, ry], 'tile_size': [tw, th]})
    return placements
