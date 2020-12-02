# Deploy BERT for Sentiment Analsysi with FastAPI

Deploy a pre-trained SBERT model for topic trakcing and detection

## Demo

The model is trained to assign topics to the articles.

```js
{
    "article_uid": "b3i1up4btj34gtp2fv54yi2f45hgf2jh4c5hv254",
    "emebedding": "[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]"
}
```

## Installation

```

Download the pre-trained model:

```sh
#TODO: Add model downloading scripts for UMAP and SBERT.
```

## Test the setup

Start the HTTP server:

```sh
bin/start_server

docker run  --env-file ./env topic-thunder

docker build -t topic-thunder . && docker run --env-file ./.env -p 8080:8080 topic-thunder 
docker-compose 

```

Send a test request:

```sh
bin/test_request
```

## License

MIT
