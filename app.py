import ssl
import certifi

# Fix SSL for macOS / broken certs
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from signLanguage.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()
