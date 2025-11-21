from PIL import Image

def crop_normalized_bbox(img: Image.Image, bbox: list[float]):
    """
    img: PIL.Image opened image
    bbox: list [x, y, w, h], normalized 0-1
    returns cropped PIL.Image
    """
    W, H = img.size
    x, y, w, h = bbox

    left   = int(x * W)
    top    = int(y * H)
    right  = int((x + w) * W)
    bottom = int((y + h) * H)

    return img.crop((left, top, right, bottom))

def crop_normalized_bbox_square(img: Image.Image, bbox: list[float]):
    """
    img: PIL.Image opened image
    bbox: list [x, y, w, h], normalized 0-1
    returns cropped PIL.Image as square
    """
    W, H = img.size
    x, y, w, h = bbox

    # Convert normalized bbox to pixel coords
    left   = int(x * W)
    top    = int(y * H)
    right  = int((x + w) * W)
    bottom = int((y + h) * H)

    # Original width/height in pixels
    bw = right - left
    bh = bottom - top

    # Determine square side
    side = max(bw, bh)
    
    # Compute center of bbox
    cx = left + bw // 2
    cy = top + bh // 2

    # Recompute square boundaries
    half = side // 2
    new_left   = cx - half
    new_top    = cy - half
    new_right  = new_left + side
    new_bottom = new_top + side

    # Clamp to image boundaries
    new_left   = max(0, new_left)
    new_top    = max(0, new_top)
    new_right  = min(W, new_right)
    new_bottom = min(H, new_bottom)

    return img.crop((new_left, new_top, new_right, new_bottom))
