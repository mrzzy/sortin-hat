#
# Sortin-hat
# Docker Container
# Ray Cluster
#

FROM rayproject/ray:2.0.0-py38-cpu

# install pip module requirements
RUN pip install --no-cache-dir scikit-learn==1.1.2 mlflow==1.29.0

# copy pipeline source code
ENV PYTHONPATH="/src:$PYTHONPATH"
COPY pipeline/src /src
