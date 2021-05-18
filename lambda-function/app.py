try:
    import unzip_requirements
except ImportError:
    pass

import io
import os
import shutil
import logging
import boto3
import tensorflow as tf

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_KEY = os.environ['MODEL_KEY']
MODEL_PATH = os.path.join('/tmp', MODEL_KEY)

s3_client = None
model = None

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def download_dir(client, bucket, directory, local):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(
            Bucket=bucket, Delimiter='/', Prefix=directory):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, bucket, subdir.get('Prefix'), local)
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                key = file.get('Key')
                local_path = os.path.join(local, key)

                if not os.path.exists(os.path.dirname(local_path)):
                    os.makedirs(os.path.dirname(local_path))
                client.download_file(bucket, key, local_path)


def handler(event, _context):
    logger.info(f'Event: {event}')

    global s3_client, model
    if s3_client is None:
        session = boto3.Session()
        s3_client = session.client('s3')

    if model is None:
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        download_dir(s3_client, MODEL_BUCKET, MODEL_KEY, '/tmp')
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
