#
# Sortin-hat
# Docker Container
# MLFlow
#

FROM python:3.8-slim

RUN pip install --no-cache-dir mlflow==1.29.0 psycopg2-binary==2.9.4 google-cloud-storage==1.44.0

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8889"]
