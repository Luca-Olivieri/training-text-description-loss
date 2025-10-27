# 1. Define parameters

# Paths
PRIVATE_DATASETS_PATH=/megaverse/datasets/$(whoami)
SHARED_DATASETS_PATH=/multiverse/datasets/shared

# Ollama
OLLAMA_MODELS_PATH=${SHARED_DATASETS_PATH}/ollama_models
OLLAMA_NUM_PARALLEL=1
OLLAMA_DEBUG=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_MAX_QUEUE=512
OLLAMA_KEEP_ALIVE=30m

# 2. Write all parameters to docker.env

cat > docker.env <<EOF
USER=$(whoami)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
PROJECT_FOLDER=..
PRIVATE_DATASETS_PATH=${PRIVATE_DATASETS_PATH}
SHARED_DATASETS_PATH=${SHARED_DATASETS_PATH}
MEMORY_LIMIT="32g"
PYTHONPATH=/home/${USER}/exp/src
GPU_DEVICE=${GPU_DEVICE}
CPU_SET=${CPU_SET}
TTDL_IMAGE_NAME=${TTDL_IMAGE_NAME}
TTDL_TAG=${TTDL_TAG}
TTDL_CONTAINER_NAME="${USER}_pyapp_GPU${GPU_DEVICE//,/.}_CPU${CPU_SET}"
OLLAMA_MODELS_PATH=${OLLAMA_MODELS_PATH}
OLLAMA_TAG=${OLLAMA_TAG}
OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}
OLLAMA_DEBUG=${OLLAMA_DEBUG}
OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}
OLLAMA_MAX_QUEUE=${OLLAMA_MAX_QUEUE}
OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}
OLLAMA_CONTAINER_NAME="${USER}_ollama_GPU${GPU_DEVICE//,/.}_CPU${CPU_SET}"
NETWORK_CONTAINER_NAME="${USER}_network_GPU${GPU_DEVICE//,/.}_CPU${CPU_SET}"
EOF

echo ${POSTFIX}

PROJECT_NAME="${USER}-ttdl-proj-gpu${GPU_DEVICE//,/_}-cpu${CPU_SET}"

# 3. Execute 'docker compose' UP or DOWN

# Read action from the first argument, default to "up"
ACTION=${1:-""}

if [ "$ACTION" = "up" ]; then
    echo "Bringing containers up..."
    docker compose --env-file ./docker.env -p "${PROJECT_NAME}" up -d
elif [ "$ACTION" = "down" ]; then
    echo "Stopping containers..."
    docker compose --env-file ./docker.env -p "${PROJECT_NAME}" down
else
    echo "Invalid action: $ACTION. Use 'up' or 'down'."
    exit 1
fi
