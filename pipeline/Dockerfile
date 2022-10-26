#
# Sortin-hat
# Docker Container
# Airflow Pipeline
#

FROM apache/airflow:2.4.1-python3.8

# install pip module requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
