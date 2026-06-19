# hash:sha256:c55e4ebbcb20c9ff9003dcb164159f6bc8b7617975fc0c4513ab32346b5bbc3f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    boto3==1.43.33 \
    mlflow==3.14.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
