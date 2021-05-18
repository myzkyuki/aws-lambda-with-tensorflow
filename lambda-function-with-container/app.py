import io
import logging
import boto3
import tensorflow as tf

MODEL_PATH = 'saved_model'

s3_client = None
model = None

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, _context):
    logger.info(f'Event: {event}')

    global s3_client, model
    if s3_client is None:
        session = boto3.Session()
        s3_client = session.client('s3')

    if model is None:
        model = tf.saved_model.load(MODEL_PATH)

    # Download image
    s3_object = s3_client.get_object(
        Bucket=event['bucket'], Key=event['filename'])
    image = io.BytesIO(s3_object['Body'].read())
    image = tf.io.decode_jpeg(image.getvalue())

    # Load model
    serve = model.signatures['serving_default']
    result = serve(tf.expand_dims(image, axis=0))

    # Format result
    labels = result['labels'][0].numpy()
    confidences = result['confidences'][0].numpy()
    result = [{'class': l.decode(), 'confidence': float(c)}
              for l, c in zip(labels, confidences)]

    logger.info(f'Result: {result}')

    return result
