import os
import shutil
from tools.preference_file_handler import PreferenceFileHandler
from deepface import DeepFace
import os
import numpy as np
def compute_sort():
    """
    The sort action refers to create a directory for each of the faces recognized
    within the images under the provided directory and fill each of the created
    directories with the images that contain the respective face.
    """

    models = ["VGG-Face","Facenet","Facenet512","OpenFace","DeepFace","DeepID","ArcFace","Dlib","SFace"]
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    backends = ['opencv','ssd','dlib','mtcnn','retinaface','mediapipe']

    input_dir = PreferenceFileHandler.get_base_directory()
    output_dir = PreferenceFileHandler.get_output_directory()

    image_file_extensions = [".jpeg", ".jpg", ".png"]

    # Get a list of tuples with the path and names of the images in the input directory
    with os.scandir(input_dir) as entries:
        images = [(entry.path, entry.name) for entry in entries if os.path.splitext(entry.path)[1] in image_file_extensions]

    # Non elegant name of producing dir names
    faces_discovered_counter = 0

    for image_path, image_name in images:
        # TODO
        # Face detection
        status, path, distance = face_detection(image_path, output_dir, models[2], metrics[2])
        if status == 0:
            new_path = f"{output_dir}/face_{faces_discovered_counter}"
            faces_discovered_counter += 1

            # Directory creation
            os.mkdir(new_path)

            # Moving the file to the new directory
            output_path = f"{new_path}/{image_name}"
            # Copy the file
            shutil.copyfile(image_path, output_path)

        else:
            output_path = f"{os.path.dirname(path)}/{image_name}"
            shutil.copyfile(image_path, output_path)


def face_detection(path, output_dir, model, metric):

    if (len(os.listdir(output_dir)) == 0):
        return 0, 0, 0

    aux_output_dir = output_dir + '/'
    df = DeepFace.find(img_path = path, db_path = aux_output_dir, model_name=model, distance_metric = metric, prog_bar = False, enforce_detection=False, silent = True)
    representations_path = f"{output_dir}/representations_facenet512.pkl"
    os.remove(representations_path)

    if df.empty:
        return 0, 0, 0
    else:
        print(df.to_string())
        return 1, df.at[0, 'identity'], df.at[0, 'Facenet512_euclidean_l2']