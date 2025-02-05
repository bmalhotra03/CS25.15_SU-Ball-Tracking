import asyncio
import numpy as np

async def left_or_right(frame, coordinates):
    frame_horizontal_center = frame.shape[1] / 2
    frame_vertical_center = frame.shape[0] / 2

    object_center_x = (coordinates[0] + coordinates[2]) / 2
    object_center_y = (coordinates[1] + coordinates[3]) / 2

    if object_center_x < frame_horizontal_center and object_center_y < frame_vertical_center:
        return 'top-left'
    elif object_center_x >= frame_horizontal_center and object_center_y < frame_vertical_center:
        return 'top-right'
    elif object_center_x < frame_horizontal_center and object_center_y >= frame_vertical_center:
        return 'bottom-left'
    else:
        return 'bottom-right'
    
async def closest_camera(frame, coordinates):
    closest_camera_index = 0
    closest_area = 0

    for i in range(len(coordinates)):
        for j in range(len(coordinates[i])):
            area = (coordinates[i][j][2] - coordinates[i][j][0]) * (coordinates[i][j][3] - coordinates[i][j][1])
            if area > closest_area:
                closest_area = area
                closest_camera_index = i

    return closest_camera_index
