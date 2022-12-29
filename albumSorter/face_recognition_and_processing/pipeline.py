import os
import shutil
from tools.preference_file_handler import PreferenceFileHandler


def compute_sort():
    """
    The sort action refers to create a directory for each of the faces recognized
    within the images under the provided directory and fill each of the created
    directories with the images that contain the respective face.
    """
    input_dir = PreferenceFileHandler.get_base_directory()
    output_dir = PreferenceFileHandler.get_output_directory()

    image_file_extensions = [".jpeg", ".jpg", ".png"]

    # Get a list of tuples with the path and names of the images in the input directory
    with os.scandir(input_dir) as entries:
        images = [(entry.path, entry.name) for entry in entries if os.path.splitext(entry.path)[1] in image_file_extensions]

    # Dictionary with 
    recognized_faces = dict()

    # Non elegant name of producing dir names
    faces_discovered_counter = 0

    for image_path, image_name in images:
        # TODO
        # Face detection
        face_detection(image_path)

        # Embeddings
        detected_embeddings = "(^w^)"

        new_face_detected = True
        # Iterate over recognized faces
        for key_embedding, path in recognized_faces.items():
            # Condition to determine if the faces has already been detected
            cond = key_embedding > detected_embeddings
            if not cond:
                continue

            # Construct output path
            output_path = f"{path}/{image_name}"
            # Copy the file
            shutil.copyfile(image_path, output_path)
            # Mark the flag
            new_face_detected = False

        # New face detected
        if new_face_detected:
            # New path creation
            new_path = f"{output_dir}/face_{faces_discovered_counter}"
            faces_discovered_counter += 1

            # Directory creation
            os.mkdir(new_path)

            # Moving the file to the new directory
            output_path = f"{new_path}/{image_name}"
            # Copy the file
            shutil.copyfile(image_path, output_path)

            # Add the entry to the dictionary
            recognized_faces[detected_embeddings] = new_path

    print(f"\n\nImages in folder:\n{images}")


def face_detection(path):
    return