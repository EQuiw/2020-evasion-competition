FROM python:3.7-slim

#############################
# INSTALL PYTHON DEPENDENCIES
#############################

# install git for pip install git+https://
RUN apt-get -o Acquire::Max-FutureTime=100000 update \
 && apt-get install -y --no-install-recommends build-essential git 

# ssdeep dependencies
# RUN apt-get install libffi-dev \
#    python3 python3-dev python3-pip libfuzzy-dev libfuzzy2 python3-setuptools \
#    automake autoconf libtool

# create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy and install python requirements + ember from github
COPY docker-requirements.txt .
COPY requirements.txt .

# if ssdeep shall be used
# RUN BUILD_LIB=1 pip install ssdeep>=3.4

RUN pip install --no-cache-dir -r docker-requirements.txt \
 && pip install --no-cache-dir git+https://github.com/endgameinc/ember.git \
 && pip install --no-cache-dir -r requirements.txt

#############################
# REBASE & DEPLOY CODE
#############################

# rebase to make a smaller image
FROM python:3.7-slim

# required libgomp1 for ember
RUN apt-get -o Acquire::Max-FutureTime=100000 update \
    && apt-get -y --no-install-recommends install \
        libgomp1 upx \
    && rm -rf /var/lib/apt/lists/*

# copy python virtual env (all dependencies) from previous image
COPY --from=0 /opt/venv /opt/venv

# copy defender code to /opt/defender
COPY defender /opt/defender

#############################
# SETUP ENVIRONMENT
#############################

# open port 8080
EXPOSE 8080

# add a defender user and switch user
RUN groupadd -r defender && useradd --no-log-init -r -g defender defender
USER defender

# change working directory
WORKDIR /opt/defender/

# update environmental variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/defender"

ENV MODEL_CONFIG_FILE models/ensemble.yml

#############################
# RUN CODE
#############################
CMD ["python","__main__.py"]

## TO BUILD IMAGE:
# docker build -t ember .
## TO RUN IMAGE (ENVIRONMENTAL VARIABLES DECLARED ABOVE)
# docker run -itp 8080:8080 ember
## TO RUN IMAGE (OVERRIDE ENVIRONMENTAL VARIABLES DECLARED ABOVE)
# docker run -itp 8080:8080 --env DF_MODEL_GZ_PATH="models/ember_model.txt.gz" --env DF_MODEL_THRESH=0.8336 --env DF_MODEL_NAME=myember ember
