import os
import json
import argparse
import tensorflow as tf

MEAN_RGB = [103.939, 116.779, 123.68]
IMAGENET_CLASS_URI = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'


class ModelExporter:
    def __init__(self, top=5):
        self.top = top
        self.classes = self.load_classes()

    @staticmethod
    def resnet50_preprocess(image, input_shape):
        image = tf.image.resize(image, input_shape)
        image = image[..., ::-1]
        image -= MEAN_RGB
        return image

    @staticmethod
    def load_classes():
        file_path = tf.keras.utils.get_file(
            os.path.basename(IMAGENET_CLASS_URI), IMAGENET_CLASS_URI)
        with open(file_path) as f:
            classes = json.load(f)

        classes = [classes[str(i)][1] for i in range(len(classes))]
        classes = tf.convert_to_tensor(classes)

        return classes

    def build_serve_fn(self, model):
        @tf.function(
            input_signature=[tf.TensorSpec([None, None, None, 3], dtype=tf.uint8)])
        def serve(images):
            images = self.resnet50_preprocess(images, model.input.shape[1:3])
            predictions = model(images)
            confidences, labels = tf.math.top_k(predictions, k=self.top)
            labels = tf.map_fn(
                lambda x: tf.map_fn(
                    lambda y: self.classes[y], x,
                    dtype=tf.string),
                labels, dtype=tf.string)

            return {'labels': labels, 'confidences': confidences}

        return serve

    def export(self, model, export_path):
        serve_fn = self.build_serve_fn(model)
        tf.saved_model.save(model, export_path, signatures={
            'serving_default': serve_fn})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_path', type=str, required=True)
    args = parser.parse_args()

    # Build model
    model = tf.keras.applications.ResNet50(weights='imagenet')

    # Export model
    exporter = ModelExporter()
    exporter.export(model, args.export_path)
