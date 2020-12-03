from sentence_transformers import SentenceTransformer, models
import torch
import boto3
import os
import tarfile, pickle
import io
import base64
import json
import re

s3 = boto3.client('s3',)


class SBERT:
    def __init__(self, model_path=None, s3_bucket=None, file_prefix=None):
        # Apply mean pooling to get one fixed sized sentence vector
        self.model = self.from_pretrained()

    def from_pretrained(self, model_path="T-Systems-onsite/bert-german-dbmdz-uncased-sentence-stsb"):
        word_embedding_model = models.Transformer(model_path, max_seq_length=512)

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        # join BERT model and pooling to get the sentence transformer
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return model

    def load_model_from_s3(self, model_path: str, s3_bucket: str, file_prefix: str):

        ### Future method which implements loading the pickled model from S3 ####

        if model_path and s3_bucket and file_prefix:
            obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
            # self.model, self.tokenizer = self.from_pretrained(model_path, s3_bucket, file_prefix)

            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            body_string = obj['Body'].read()
            model = pickle.loads(body_string)

            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def encode(self, text_array):
        encoded = self.model.encode(text_array, show_progress_bar=True, batch_size=32)
        return encoded
