# AWS Lambda with TensorFlow
ここは、AWS Lambda上でTensorFlowを動かすサンプルのリポジトリです。

## 共通
### Serverless Frameworkのインストール
```
$ npm install -g serverless
$ npm install --save serverless-python-requirements
```

### Saved modelの出力
```bash
$ python model_exporter.py \
    --export_path lambda-function-with-container/saved_model
```

## コンテナの使用したLambda function
### コンテナイメージの作成
```bash
$ cd lambda-function-with-container
$ docker build -t lambda-imagenet-resnet50 .
```

### ECRにpush
```bash
$ aws ecr create-repository \
    --repository-name lambda-imagenet-resnet50
$ aws ecr get-login-password --region <Region> | \
    docker login --username AWS --password-stdin <Account>.dkr.ecr.<Region>.amazonaws.com
$ docker tag lambda-imagenet-resnet50:latest \
    <Account>.dkr.ecr.<Region>.amazonaws.com/lambda-imagenet-resnet50:latest
$ docker push <Account>.dkr.ecr.<Region>.amazonaws.com/lambda-imagenet-resnet50:latest
```

### デプロイ
```bash
$ serverless deploy \
    --imageUrl <Account>.dkr.ecr.<Region>.amazonaws.com/lambda-imagenet-resnet50:latest
```

### 推論
```bash
$ serverless invoke \
    --function classify \
    --data '{"bucket": "<Bucket>", "filename": "<S3 Key>"}'
```

## 通常のLambda function
### S3にモデルをアップロード
```bash
$ aws s3 cp saved_model s3://<Bucket>/<Model key> --recursive
```

###  デプロイ
```bash
$ cd lambda-layer
$ serverless deploy
$ cd ../lambda-function
$ serverless deploy --modelBucket <Bucket> --modelKey <S3 key>
```

### 推論
```bash
$ serverless invoke \
    --function classify \
    --data '{"bucket": "<Bucket>", "filename": "<S3 Key>"}'
```