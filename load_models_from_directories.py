import joblib, os

def load_models_from_directories(base_path):
    """
    Load all trained regression models (one model for each cross-validation split)
    Load 'model.pkl' from each directory within the base_path.

    :param base_path: The parent directory containing split subdirectories.
    :return: A dictionary where keys are directory names and values are the loaded models.
    """

    models = {}
    for name in os.listdir(base_path):
        dir_path = os.path.join(base_path, name)
        if os.path.isdir(dir_path):
            model_path = os.path.join(dir_path, 'model.pkl')
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"'model.pkl' not found in {dir_path}")

    return models

