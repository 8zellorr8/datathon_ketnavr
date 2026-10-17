"""Pack chronological frames without changing original evidence indices.

Only generated JPEG bytes and a panel map are returned. No source image or
description is logged, uploaded, written to disk, or removed from the timeline.
"""
from __future__ import annotations

import base64
from io import BytesIO
import math
import time


def pack_contact_sheets(frames, *, frames_per_sheet=6, max_side=1250, deadline=None):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if type(frames_per_sheet) is not int or not 4 <= frames_per_sheet <= 6:
        raise ValueError('frames_per_sheet must be an integer between 4 and 6')
    if type(max_side) is not int or not 768 <= max_side <= 1600:
        raise ValueError('contact_sheet_max_side must be an integer between 768 and 1600')
    if not frames:
        raise ValueError('Contact sheets require at least one original frame')

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Contact-sheet preparation exceeded its time budget')

    try:
        font = ImageFont.truetype('arial.ttf', 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    images, mapping, dimensions = [], [], []
    padding, gap, header = 12, 10, 36
    for start in range(0, len(frames), frames_per_sheet):
        check_deadline()
        group = frames[start:start + frames_per_sheet]
        columns = 3 if len(group) > 4 else min(2, len(group))
        rows = math.ceil(len(group) / columns)
        sheet_width = min(max_side, 960)
        cell_width = (sheet_width - 2 * padding - (columns - 1) * gap) // columns
        natural_height = 0
        for frame in group:
            with Image.open(BytesIO(base64.b64decode(frame['image_base64'], validate=True))) as source:
                natural_height = max(natural_height, math.ceil(cell_width * source.height / source.width))
        # Use the same coordinate system for the panel map and rendered labels.
        cell_height = min(header + natural_height, (max_side - 2 * padding - (rows - 1) * gap) // rows)
        image_height = cell_height - header
        sheet_height = 2 * padding + rows * cell_height + (rows - 1) * gap
        sheet = Image.new('RGB', (sheet_width, sheet_height), '#e5e7eb')
        draw = ImageDraw.Draw(sheet)
        for offset, frame in enumerate(group):
            check_deadline()
            index = start + offset
            row, column = divmod(offset, columns)
            x = padding + column * (cell_width + gap)
            y = padding + row * (cell_height + gap)
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), fill='#111827')
            draw.text((x + 6, y + 5), f'#{index} t={frame["timestamp_sec"]:.6f}s', font=font, fill='white')
            try:
                with Image.open(BytesIO(base64.b64decode(frame['image_base64'], validate=True))) as source:
                    if source.width * source.height > 16 * 1024 * 1024:
                        raise ValueError('Decoded image resolution exceeds the packing budget')
                    oriented = ImageOps.exif_transpose(source).convert('RGB')
                    thumbnail = ImageOps.contain(oriented, (cell_width, image_height), Image.Resampling.LANCZOS)
                    image_x = x + (cell_width - thumbnail.width) // 2
                    image_y = y + header + (image_height - thumbnail.height) // 2
                    sheet.paste(thumbnail, (image_x, image_y))
                    image_bbox = [image_x, image_y, thumbnail.width, thumbnail.height]
                    thumbnail.close()
                    oriented.close()
            except (OSError, ValueError) as exc:
                raise ValueError(f'Cannot decode original frame {index} for local packing') from exc
            mapping.append({
                'frame_index': index, 'timestamp_sec': frame['timestamp_sec'],
                'sheet_index': len(images), 'row': row, 'column': column,
                'panel_bbox_xywh': [x, y, cell_width, cell_height],
                'image_bbox_xywh': image_bbox,
            })
        encoded = BytesIO()
        sheet.save(encoded, format='JPEG', quality=90, optimize=True)
        sheet.close()
        images.append(base64.b64encode(encoded.getvalue()).decode('ascii'))
        dimensions.append({'width': sheet_width, 'height': sheet_height})
        check_deadline()
    return {'images': images, 'frame_map': mapping, 'sheet_dimensions': dimensions}
