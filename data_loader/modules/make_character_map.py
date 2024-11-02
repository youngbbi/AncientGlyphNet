import numpy as np
import cv2

class MakeCharacterMap:
    """
    Make character level ground truth for each character in the text polygons.
    """
    def __init__(self, min_text_size=8):
        self.min_text_size = min_text_size

    def __call__(self, data: dict) -> dict:
        """
        Create a character map for each character in the text polygons.
        :param data: {'img':, 'text_polys':, 'texts':, 'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['text_polys']
        texts = data['texts']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        character_map = np.zeros((h, w), dtype=np.float32)
        for i, (polygon, text) in enumerate(zip(text_polys, texts)):
            if ignore_tags[i] or len(text) == 0:
                continue
            for char_poly in self.split_to_characters(polygon, text):
                cv2.fillPoly(character_map, [char_poly.astype(np.int32)], 1)
        data['character_map'] = character_map
        return data

    def split_to_characters(self, polygon, text):
        """
        Split the text polygon into individual character polygons.
        :param polygon: The polygon enclosing the entire piece of text.
        :param text: The text string inside the polygon.
        :return: A list of polygons, one for each character in the text.
        """
        # Simplest approach - split the text box into equal parts
        if len(text) <= 1:
            return [polygon]

        x_coords = np.linspace(polygon[0, 0], polygon[2, 0], num=len(text) + 1)
        height_top = polygon[0, 1]
        height_bottom = polygon[2, 1]
        char_polygons = []
        for i in range(len(text)):
            char_poly = np.array([
                [x_coords[i], height_top],
                [x_coords[i + 1], height_top],
                [x_coords[i + 1], height_bottom],
                [x_coords[i], height_bottom]
            ])
            char_polygons.append(char_poly)
        return char_polygons

# Example usage
if __name__ == '__main__':
    data = {
        'img': np.zeros((100, 100, 3), dtype=np.uint8),  # Example image
        'text_polys': [np.array([[10, 10], [50, 10], [50, 50], [10, 50]])],  # Example polygon
        'texts': ['TEXT'],  # Corresponding text
        'ignore_tags': [False]  # Whether to ignore this text
    }
    make_char_map = MakeCharacterMap()
    data = make_char_map(data)
    char_map = data['character_map']
    # You can now save or visualize the character_map as needed.
