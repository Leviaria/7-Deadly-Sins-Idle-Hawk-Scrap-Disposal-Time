from loguru import logger
import subprocess
from PIL import Image, ImageDraw
import cv2
import numpy as np
import threading

def take_screenshot(output_path='screenshot.png'):
    logger.info(f"Taking screenshot and saving to {output_path}")
    subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=open(output_path, 'wb'))

def draw_grid(image_path, grid_width, grid_height, grid_x, grid_y):
    logger.info(f"Drawing grid on image {image_path}")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    cell_width = grid_width // 8
    cell_height = grid_height // 10

    for i in range(9):
        x = grid_x + i * cell_width
        draw.line([(x, grid_y), (x, grid_y + grid_height)], fill="red", width=2)

    for j in range(11):
        y = grid_y + j * cell_height
        draw.line([(grid_x, y), (grid_x + grid_width, y)], fill="red", width=2)

    return cell_width, cell_height

def template_matching(gray_image, templates, cell_width, cell_height, grid_x, grid_y):
    logger.info("Performing template matching")
    result_grid = []
    for row in range(10):
        result_row = []
        for col in range(8):
            top_left_x = grid_x + col * cell_width
            top_left_y = grid_y + row * cell_height
            bottom_right_x = top_left_x + cell_width
            bottom_right_y = top_left_y + cell_height

            cell = gray_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            matched = False
            for name, template in templates.items():
                result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > 0.8:
                    result_row.append(name)
                    matched = True
                    break

            if not matched:
                result_row.append('leer')

        result_grid.append(result_row)

    return result_grid

def find_and_click_adjacent_fruits(result_grid, cell_width, cell_height, grid_x, grid_y):
    logger.info("Finding and clicking adjacent fruits")
    clicked = False
    clicked_fruits = set()

    for row in range(10):
        for col in range(8):
            if result_grid[row][col] == 'leer':
                fruit_count = {}
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r, c = row + dr, col + dc

                    while 0 <= r < 10 and 0 <= c < 8 and result_grid[r][c] == 'leer':
                        r += dr
                        c += dc

                    if 0 <= r < 10 and 0 <= c < 8 and result_grid[r][c] != 'leer':
                        fruit = result_grid[r][c]
                        if fruit in fruit_count:
                            fruit_count[fruit] += 1
                        else:
                            fruit_count[fruit] = 1

                for fruit, count in fruit_count.items():
                    if count >= 2 and fruit not in clicked_fruits:
                        click_position_x = grid_x + col * cell_width + cell_width // 2
                        click_position_y = grid_y + row * cell_height + cell_height // 2
                        click_on_position(click_position_x, click_position_y)
                        logger.info(f"Clicked on position ({click_position_x}, {click_position_y}) to collect fruit: {fruit}")
                        clicked_fruits.add(fruit)
                        clicked = True
                        break

    return clicked

def click_on_position(x, y):
    logger.info(f"Clicking on position ({x}, {y})")
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])

def main():
    templates = {}
    template_files = ['images/apple.png', 'images/avocado.png', 'images/pear.png', 'images/bread.png', 'images/pizza.png', 'images/steak.png', 'images/grape.png']
    for template_name in template_files:
        template_image = cv2.imread(template_name, 0)
        if template_image is None:
            logger.error(f"Error: Template {template_name} could not be loaded. Please check the path.")
            return
        templates[template_name.split('/')[-1].split('.')[0]] = template_image

    grid_width = 700
    grid_height = 740
    grid_x = 10
    grid_y = 415

    while True:
        screenshot_thread = threading.Thread(target=take_screenshot)
        screenshot_thread.start()
        screenshot_thread.join()

        image_path = 'screenshot.png'
        image = cv2.imread(image_path, 0)
        if image is None:
            logger.error(f"Error: Screenshot {image_path} could not be loaded.")
            return

        cell_width, cell_height = draw_grid(image_path, grid_width, grid_height, grid_x, grid_y)
        result_grid = template_matching(image, templates, cell_width, cell_height, grid_x, grid_y)

        clicked = find_and_click_adjacent_fruits(result_grid, cell_width, cell_height, grid_x, grid_y)

        if not clicked:
            continue

if __name__ == "__main__":
    main()
