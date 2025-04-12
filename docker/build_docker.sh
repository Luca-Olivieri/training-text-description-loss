#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")
USERNAME=$(whoami)
TAG=latest # default tag
DOCKER_FILENAME=dockerfile
IMAGE_NAME=""

while [[ $# -gt 0 ]]
do key="$1"

# argument parsing
case $key in
	-im|--image_name)
	IMAGE_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-t|--tag)
	TAG="$2"
	shift # past argument
	shift # past value
	;;
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name	name of the docker image (mandatory)"
	echo "	-t, --tag			image tag name (default 'latest')"
	exit
	;;
	*)
	echo "Wrong option(s) is selected. Use -h, --help for more information."
	exit
	;;
esac
done

# Check if IMAGE_NAME is set, if not, exit with an error
if [ -z "$IMAGE_NAME" ]; then
  echo "Error: IMAGE_NAME is required. Use -im <image_name> to provide it."
  exit 1
fi

# Prefix IMAGE_NAME with the username
IMAGE_NAME="${USERNAME}/${IMAGE_NAME}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"

docker build -t ${IMAGE_NAME}:${TAG} \
	-f ${SCRIPT_DIR}/${DOCKER_FILENAME} \
	${SCRIPT_DIR}
