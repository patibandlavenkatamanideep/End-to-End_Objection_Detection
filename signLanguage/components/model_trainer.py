import os
import sys
import yaml
import shutil
import subprocess
import glob
import ssl
import certifi

from signLanguage.utils.main_utils import read_yaml_file
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifact_entity import ModelTrainerArtifact

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def unzip_dataset(self, zip_path: str, extract_to: str):
        """Safely unzip dataset if it exists."""
        try:
            if not os.path.exists(zip_path):
                logging.warning(f"{zip_path} not found. Skipping unzip.")
                return
            logging.info(f"Unzipping dataset: {zip_path}")
            shutil.unpack_archive(zip_path, extract_to)
            os.remove(zip_path)
            logging.info(f"Dataset extracted to {extract_to}")
        except Exception as e:
            raise SignException(e, sys)

    def find_data_yaml(self) -> str:
        """Find the latest data.yaml file dynamically in artifacts."""
        try:
            artifact_dirs = glob.glob(os.path.join("artifacts", "*"))
            if not artifact_dirs:
                raise FileNotFoundError("No artifact folders found in 'artifacts/' directory.")

            latest_artifact = max(artifact_dirs, key=os.path.getmtime)
            data_yaml_path = os.path.join(latest_artifact, "data_ingestion", "feature_store", "data.yaml")

            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

            logging.info(f"Found data.yaml: {data_yaml_path}")
            return data_yaml_path

        except Exception as e:
            raise SignException(e, sys)

    def prepare_model_config(self, weight_name: str, num_classes: int) -> str:
        """Create custom YOLOv5 config for training."""
        try:
            model_name = os.path.splitext(weight_name)[0]
            config_path = f"yolov5/models/{model_name}.yaml"

            config = read_yaml_file(config_path)
            config['nc'] = num_classes

            custom_config_path = f"yolov5/models/custom_{model_name}.yaml"
            with open(custom_config_path, 'w') as f:
                yaml.dump(config, f)

            logging.info(f"Custom YOLOv5 config created: {custom_config_path}")
            return custom_config_path

        except Exception as e:
            raise SignException(e, sys)

    def train_yolov5(self, custom_config_path: str, data_yaml_path: str):
        """Train YOLOv5 using subprocess and save best model."""
        try:
            cmd = [
                sys.executable,  # current python interpreter
                "yolov5/train.py",
                "--img", "416",
                "--batch", str(self.model_trainer_config.batch_size),
                "--epochs", str(self.model_trainer_config.no_epochs),
                "--data", data_yaml_path,
                "--cfg", custom_config_path,
                "--weights", self.model_trainer_config.weight_name,
                "--name", "yolov5_results",
                "--cache"
            ]

            logging.info(f"Running YOLOv5 training: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # Copy best.pt to model trainer directory
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            runs_train_path = "yolov5/runs/train/"
            latest_run = max(glob.glob(os.path.join(runs_train_path, "yolov5*")), key=os.path.getmtime)
            best_model_src = os.path.join(latest_run, "weights/best.pt")
            best_model_dst = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")

            if not os.path.exists(best_model_src):
                raise SignException(f"Training did not produce {best_model_src}", sys)

            shutil.copy(best_model_src, best_model_dst)
            logging.info(f"Trained model saved to {best_model_dst}")
            return best_model_dst

        except Exception as e:
            raise SignException(e, sys)

    def clean_up(self):
        """Clean temporary folders after training."""
        try:
            for folder in ["yolov5/runs", "train", "test", "data.yaml"]:
                if os.path.exists(folder):
                    if os.path.isdir(folder):
                        shutil.rmtree(folder)
                    else:
                        os.remove(folder)
            logging.info("Temporary files cleaned up.")
        except Exception as e:
            raise SignException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Full model trainer pipeline."""
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Step 1: Unzip dataset
            self.unzip_dataset("Sign_language_data.zip", ".")

            # Step 2: Load num_classes
            data_yaml_path = self.find_data_yaml()
            with open(data_yaml_path, 'r') as f:
                num_classes = yaml.safe_load(f)['nc']

            # Step 3: Prepare custom config
            custom_config_path = self.prepare_model_config(self.model_trainer_config.weight_name, num_classes)

            # Step 4: Train YOLOv5
            trained_model_path = self.train_yolov5(custom_config_path, data_yaml_path)

            # Step 5: Cleanup
            self.clean_up()

            # Step 6: Create artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=trained_model_path)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
