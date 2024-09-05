from typing import List


def extract_face_boxes(result, frame_image) -> List:
    face_boxes = []
    frame_height, frame_width = frame_image.shape[:2]

    for box in result.boxes:
        face_box = __tosquare(box.xyxy[0], frame_height, frame_width)  #  Returns left top corner's xy and wh
        if face_box is not None:
            face_box = face_box[0], face_box[1], face_box[0] + \
                face_box[2], face_box[1] + face_box[3]  #  xyxy format
            face_boxes.append(face_box)

    return face_boxes


def __tosquare(bbox, max_height, max_width):
    """
    Converts a bounding box to a square box.

    Args:
        bbox (tuple): Bounding box coordinates.
        max_height (int): Maximum height.
        max_width (int): Maximum width.

    Returns:
        tuple: Square box coordinates.
    """
    x_left_top, y_left_top, x_right_bottom, y_right_bottom = bbox
    x = int(x_left_top)
    y = int(y_left_top)
    w = int(x_right_bottom - x_left_top)
    h = int(y_right_bottom - y_left_top)

    if h > w:
        diff = h - w
        x -= diff // 2
        w += diff
    elif w > h:
        diff = w - h
        y -= diff // 2
        h += diff
    if w != h:
        pass
        # __logger.error(f"Face Box Width:{w} and Height:{h} different!!")

    x, y, w, h = __increase_squarebox_size(x, y, w, h, 0.3)

    return __ignore_face(x, y, w, h, max_height, max_width, 0.2)


def __ignore_face(x, y, w, h, max_height, max_widht, ratio):
    """
    Ignores faces that are outside the frame.

    Args:
        x (int): X-coordinate.
        y (int): Y-coordinate.
        w (int): Width.
        h (int): Height.
        max_height (int): Maximum height.
        max_widht (int): Maximum width.
        ratio (float): Ratio for threshold.

    Returns:
        tuple: New coordinates and dimensions or None if the face is to be ignored.
    """
    ignore = False
    if (
        (x < 0 and abs(x) > w * ratio)
        or (x + w > max_widht and abs(w + x - max_widht) > w * ratio)
        or (y < 0 and abs(y) > h * ratio)
        or (y + h > max_height and abs(h + y - max_height) > h * ratio)
    ):
        ignore = True
    if ignore:
        # __logger.warn(f"Face Box x:{x} and y:{y} outside of the frame!!!")
        return None
    else:
        return __handle_out_of_boundries(x, y, w, h, max_height, max_widht)


def __handle_out_of_boundries(x, y, w, h, max_height, max_width):
    """
    Handles faces that are partially outside the frame.

    Args:
        x (int): X-coordinate.
        y (int): Y-coordinate.
        w (int): Width.
        h (int): Height.
        max_height (int): Maximum height.
        max_width (int): Maximum width.

    Returns:
        tuple: New coordinates and dimensions.
    """

    # left top corner
    if x < 0 and y < 0:
        h = int(min(w - abs(x), h - abs(y)))
        x = 0
        y = 0
    # right top corner
    elif x + w > max_width and y < 0:
        h = int(min(max_width - x, h - abs(y)))
        y = 0
    # right bottom corner
    elif x + w > max_width and y + h > max_height:
        h = int(min(max_width - x, max_height - y))
    # left bottom corner
    elif x < 0 and y + h > max_height:
        h = int(min(w - abs(x), max_height - y))
        x = 0
    # left border
    elif x < 0:
        h = int(w - abs(x))
        x = 0
    # top border
    elif y < 0:
        h = int(h - abs(y))
        y = 0
    # right border
    elif x + w > max_width:
        h = int(max_width - x)
    # buttom border
    elif y + h > max_height:
        h = int(max_height - y)
    w = h
    return (x, y, w, h)


def __increase_squarebox_size(x, y, w, h, ratio):
    """
    Increases the size of a square box.

    Args:
        x (int): X-coordinate.
        y (int): Y-coordinate.
        w (int): Width.
        h (int): Height.
        ratio (float): Ratio for increase.

    Returns:
        Tuple: New coordinates and dimensions.
    """
    n_w = int(w * (1 + ratio))
    n_h = int(h * (1 + ratio))
    n_x = int(x - ((n_w - w) / 2))
    n_y = int(y - ((n_h - h) / 2))

    return (n_x, n_y, n_w, n_h)
