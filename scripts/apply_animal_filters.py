import cv2
import numpy as np



def apply_filters(face_points, image_copy_1, image_name):

    animal_filter = cv2.imread("images/" + image_name, cv2.IMREAD_UNCHANGED)

    for i in range(len(face_points)):
        # Get the width of filter depending on left and right eye brow point
        # Adjust the size of the filter slightly above eyebrow points
        filter_width = 1.1 * (face_points[i][14] + 15 - face_points[i][18] + 15)
        scale_factor = filter_width / animal_filter.shape[1]
        sg = cv2.resize(
            animal_filter,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )

        width = sg.shape[1]
        height = sg.shape[0]

        # top left corner of animal_filter: x coordinate = average x coordinate of eyes - width/2
        # y coordinate = average y coordinate of eyes - height/2
        x1 = int((face_points[i][2] + 5 + face_points[i][0] + 5) / 2 - width / 2)
        x2 = x1 + width

        y1 = int((face_points[i][3] - 65 + face_points[i][1] - 65) / 2 - height / 3)
        y2 = y1 + height

        # Create an alpha mask based on the transparency values
        alpha_fil = np.expand_dims(sg[:, :, 3] / 255.0, axis=-1)
        alpha_face = 1.0 - alpha_fil

        # Take a weighted sum of the image and the animal filter using the alpha values and (1- alpha)
        image_copy_1[y1:y2, x1:x2] = (
            alpha_fil * sg[:, :, :3] + alpha_face * image_copy_1[y1:y2, x1:x2]
        )

    return image_copy_1


