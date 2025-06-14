#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_FOLDER="$(dirname "$SCRIPT_DIR")"

EXPECTED_PARENT="/home/$(whoami)/storage"

# assert that this file is in "/home/$(whoami)/storage"
if [[ "$CODE_FOLDER" != "$EXPECTED_PARENT"* ]]; then
    echo "Error: CODE_FOLDER is not inside $EXPECTED_PARENT"
    exit 1
fi

echo "CODE_FOLDER is $CODE_FOLDER"

# IMAGE_NAME is not set initially
TAG=latest

STORAGE_FOLDER=/multiverse/storage/$(whoami)/
SHARED_DATASET_FOLDER=/multiverse/datasets/shared/
PRIVATE_DATASET_FOLDER=/multiverse/datasets/$(whoami)/

WORKING_DIR=/home/$(whoami)/exp 
MEMORY_LIMIT=32g

# options
INTERACTIVE=1
LOG_OUTPUT=1

while [[ $# -gt 0 ]]
do 
	key="$1"

	case $key in
		-im|--image_name)
		IMAGE_NAME="$2"
		shift
		shift
		;;
		-t|--tag)
		TAG="$2"
		shift
		shift
		;;
		-i|--interactive)
		INTERACTIVE="$2"
		shift
		shift
		;;
		-gd|--gpu_device)
		GPU_DEVICE="$2"
		shift
		shift
		;;
		-m|--memory_limit)
		MEMORY_LIMIT="$2"
		shift
		shift
		;;
		-cpu|--cpu_set)
		CPU_SET="$2"
		shift
		shift
		;;
		-h|--help)
		shift
		echo "Options:"
		echo "  -im, --image_name     name of the docker image (REQUIRED)"
		echo "	-t, --tag             image tag name (default \"tf2-gpu\")"
		echo "	-gd, --gpu_device     gpu to be used inside docker (REQUIRED)"
		echo "	-cn, --container_name name of container (default \"tf2_run\")"
		echo "	-m, --memory_limit    RAM limit (default 32g)"
		echo "	-cpu, --cpu_set       cpu ids to be used inside docker (REQUIRED)"
		exit
		;;
		*)
		echo "Wrong option(s). Use -h or --help for more information."
		exit 1
		;;
	esac
done

# Validate required parameters
if [ -z "$IMAGE_NAME" ]; then
  echo "Error: --image_name (-im) is required."
  exit 1
fi

if [ -z "$GPU_DEVICE" ]; then
  echo "Error: --gpu_device (-gd) is required."
  exit 1
fi

if [ -z "$CPU_SET" ]; then
  echo "Error: --cpu_set (-cpu) is required."
  exit 1
fi

# Prefix IMAGE_NAME with the username
# CONTAINER_NAME=$(whoami)_{$IMAGE_NAME}_$(date +%Y%m%d)
CONTAINER_NAME=$(whoami)_${IMAGE_NAME}_GPU${GPU_DEVICE//,/.}_$(date +%F_%H.%M)

echo "WORKING_DIR     = ${WORKING_DIR}"
echo "GPU_DEVICE      = ${GPU_DEVICE}"
echo "CPU_SET         = ${CPU_SET}"
echo "CONTAINER_NAME  = ${CONTAINER_NAME}"
echo "PORT            = ${PORT}"

echo "Running docker in interactive mode"

docker_args=(
	-d --rm -it
	# --gpus "device=${GPU_DEVICE}" \
	--gpus "\"device=${GPU_DEVICE}\""
	--cpuset-cpus ${CPU_SET} \
	--mount type=bind,source=${CODE_FOLDER},target=${WORKING_DIR} \
	--mount type=bind,source=${PRIVATE_DATASET_FOLDER},target=${WORKING_DIR}/data \
	-m ${MEMORY_LIMIT} \
	-w ${WORKING_DIR} \
	-e log=/home/log.txt \
	-e HOST_UID=$(id -u) \
	-e HOST_GID=$(id -g) \
	--name ${CONTAINER_NAME}
	--network olivieri_net
)

docker run ${docker_args[@]} $(whoami)/${IMAGE_NAME}:${TAG}
