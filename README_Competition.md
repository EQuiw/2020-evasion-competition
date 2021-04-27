# 2020-ms-adv-challenge

## Administrative Info

- [registration](https://mlsec.io/admin/login/)
- [rules](https://mlsec.io/tos/)
- [defender sample-repo](https://github.com/Azure/2020-machine-learning-security-evasion-competition/tree/master/defender)

## Requirements

A valid submission for the defense track consists of the following
1. a Docker image no larger than 1 GB when _uncompressed_ (`gzip` compression required for upload)
2. listens on port 8080
3. accepts `POST /` with header `Content-Type: application/octet-stream` and the contents of a PE file in the body
4. returns `{"result": 0}` for benign files and `{"result": 1}` for malicious files (bytes `POST`ed as `Content-Type: application/json`)
5. must exhibit a false positive rate of less than 1% and a false negative rate of less than 10% (checked on upload, during and after the Attacker Challenge using randomly-selected files)
6. for files up to 2**21 bytes (2 MiB), must respond in less than 5 seconds (a timeout results in a benign verdict)

## Docker deployment

- build docker image
  ```
  cd $REPO_DIR
  docker build -t adv-challenge .
  ```

- start container (deamon)
  ```
  docker run -d -p 8080:8080 --memory=1.5g --cpus=1 adv-challenge
  ```

- start container (interactive mode)
  ```
  docker run -it -p 8080:8080 --memory=1.5g --cpus=1 adv-challenge /bin/bash
  python __main__.py  # start flask app
  ```

- perform tests (second shell if container has been started interactively)
  ```
  cd $REPO_DIR
  python test/test_post_request.py
  
  python test/test_dataset.py --datasetpath /<TODO-PATH-TO>/MLSEC_2019_samples_and_variants/ --verbose
  ```

- Cleanup
  ```
  docker stop $(docker ps -aq)
  docker rm $(docker ps -aq)
  docker rmi $(docker images -aq)
  docker system prune  # cleans all dangling images
  ```
