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
    
    models = ["VGG-Face","Facenet","Facenet512","OpenFace","DeepFace","DeepID","ArcFace","SFace"]
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

    # Different faces for embeddings directory creation
    db_images = f"{output_dir}/aux-faces"
    representations_path = f"{db_images}/representations_arcface.pkl"
    os.mkdir(db_images)

    for image_path, image_name in images:
        # Face detection
        status, path, distance = face_detection(image_path, db_images, models[6], metrics[0])
        if status == 0:
            new_path = f"{output_dir}/face_{faces_discovered_counter}"
            aux_output_path = f"{db_images}/face_{faces_discovered_counter}{os.path.splitext(image_name)[1]}"
            faces_discovered_counter += 1

            # Directory creation
            os.mkdir(new_path)

            # Moving the file to the new directory
            output_path = f"{new_path}/{image_name}"
            
            # Copy the file
            shutil.copyfile(image_path, output_path)

            # Copy the file to aux-faces
            shutil.copyfile(image_path, aux_output_path)
            
            if os.path.exists(representations_path):
                os.remove(representations_path)

        else:
            base=os.path.basename(path)
            output_path = f"{output_dir}/{os.path.splitext(base)[0]}/{image_name}"
            shutil.copyfile(image_path, output_path)


def face_detection(path, output_dir, model, metric):

    if (len(os.listdir(output_dir)) == 0):
        return 0, 0, 0

    aux_output_dir = output_dir + '/'
    df = DeepFace.find(img_path = path, db_path = aux_output_dir, model_name=model, distance_metric = metric, prog_bar = False, enforce_detection=False, silent = True)

    if df.empty:
        return 0, 0, 0
    else:
        print(df.to_string())
        return 1, df.at[0, 'identity'], df.at[0, 'ArcFace_cosine']
