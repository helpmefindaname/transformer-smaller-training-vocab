# todo: use multistage build to only use what is required at the end.
FROM python:3.10.1
ARG BUILD=prod
ARG BUILD_VERSION=0.1.0
ENV POETRY_VERSION=1.3.1
# should be fixed but as high as possible

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# RUN apt-get update && apt-get install -y <ubuntu-packages> # install additional packages such as tesseract, imagemagick, g++ etc. IF REQUIRED
WORKDIR /app
CMD ["python_template"] # TODO: change to run commando

RUN curl -sSL https://install.python-poetry.org | python -

COPY . .

RUN poetry install --without dev
RUN poetry version $BUILD_VERSION

RUN if [ $BUILD = "test" ] ; then poetry install; fi



