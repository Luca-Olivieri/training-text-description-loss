This folder collects all the files required to build, launch and stop the Docker container with all its dependencies. Here is a description of the most important script for the user.

- **`build_docker.sh`** is the script building the container as defined in the `Dockerfile` and including the Python dependencies listed in the `requirements.txt`.
- **`docker-compose.yml`** is the config file defining the parameters for the experiments.
- **`docker.sh`** is the script running and stopping the container defined in the `docker-compose.yml`.
