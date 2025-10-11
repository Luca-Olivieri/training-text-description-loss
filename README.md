# Training Vision Tasks Using Diagnostic Text Description Loss

## Installation

1. Navigate to the project’s root directory:
```bash
cd <PROJECTS_ROOT_DIR>
```

2. Clone this repository:
```bash
git clone https://github.com/Luca-Olivieri/training-text-description-loss.git
```

3. Clone repositories of dependencies:
```bash
cd training-text-description-loss/src/vendors
```
```bash
git clone https://github.com/ExplainableML/flair.git
```

4. Open .py file `flair/src/flair/factory.py` (located  in `src/vendors`), and modify line 145 into this:
```python
checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
```

5. Navigate to the project docker directory:
```bash
cd ../../docker
```
  and update the `dockerfile` with your system credentials at line 8, 9, 10:
```docker
ARG USERNAME=...
ARG USER_UID=...
ARG USER_GID=...
```

6. Build the docker container:
```bash
./build_docker.sh -im ttdl -t v2
```

7. Pull the `ollama` container:
```bash
docker pull ollama/ollama:0.8.0
```

The project runs with two docker containers:

- `pyapp`, containing and executing the application logic.
- `ollama`, running Ollama VLMs.

They are run through a `docker-compose.yml` (in the `docker` directory) which spawns them and binds them so that a `pyapp` instance can only communicate to its paired `ollama` instance.

The `docker-compose.yml` file is controlled through `docker.sh` (in the same directory), which brings the containers pair up and down and propagates the environment.

Before running the containers, you have to set their environment:

- In `docker.sh`, you have the provide the path of the directories to bind in the docker runtime (private data, shared data, etc.), as well as specify the parameters to pass to the containers. In particular, you should check the Ollama models path (which is in dataset shared so it should be fine, the rest of Ollama parameters is fine for the intended experiments).
- All additional environment modifications can be made by adapting the `docker-compose.yml` file and the `docker.sh` controlling it.
- `docker.env` file should not be touched, it is generated automatically by other files.

**NOTE**: Since the pipelines require some pre-built resources (e.g., models weights, cache checkpoints, prompts, etc.), it is better to mount my private data folder alongside the others and access the resources from it.

## Usage

To bring the containers up and down, navigate to the docker directory and execute this command, defining the GPUs and CPUs to provide.

```bash
GPU_DEVICE=... CPU_SET=... TTDL_IMAGE_NAME=ttdl TTDL_TAG=v2 OLLAMA_TAG=0.8.0 ./docker.sh [up, down]
```

**NOTE**: The application logic is made to run on a single GPU. No implementation for distributed computation is provided. Ollama, instead, splits the computation on the least number of GPUs (ideally, 1) to accomodate for the vRAM. If there is not enough vRAM, it will offload part of the model on the CPU, making the inference excessively slow. The device in which the Ollama model is running can be checked by entering the Ollama container and executing `ollama ps`.

**NOTE**: Experiments with the current configurations take from ~13.5 to ~20.5 GB of vRAM.

The logic inside `pyapp` source code (in `src`) is splitted in many files. The entry point files for the training experiments are inside the `src/train` directory.

### Configurations

Experiments are heavily influenced by the configs. Each experiments is self-contained in its directory (apart from the core logic, which is shared), and it is bound to the config in the same directory.

All configs are built starting from `config/base_config.yml`, but, apart from this, they are independent.

Most likely, you will have to adapt many .yml config entries to accomodate for the model, cache and dataset paths.

Also, each .py file has a path reference to its config.yml file, hence when duplicating train scripts change the config reference to the actual new config file, or it will used the original, wrong one.

In particular, pay attention to the segmentation models and FLAIR paths so that they do not get downloaded in weird places.

Some experiments are made up of more steps. The earlier steps are contained into its `base` folder. Baselines can be built by toggling the additional `with_text` and `with_cache` flags off in the configs and adapting the starting weights and configs appropriately.

### Pre-trained Weights

Some pre-trained weights are available in my private data under the directory `torch_weights`, along with some old experiments.

The only two pre-trained weights you should need are:

- **LRASPP MobileNet v3 Large**: `<OLIVIERI_PRIVATE_PATH>/torch_weights/seg/lraspp_mobilenet_v3_large/no_text/lraspp_mobilenet_v3_large-enc-pt.pth`
- **FLAIR**: inside `<OLIVIERI_PRIVATE_PATH>/torch_weights/vle/flair` (the logic access it by folder, no need to specify the actual checkpoint file).

### Experiment Outputs

The experiments produce logs and weights as output and place it in a directory built from the configs (look up in the code how it is formed starting from root experiment folder, experiment name and variation of the experiment).

This way, all experiments results are available in one place and arranged cleanly.

Logs are available in textual and TensorBoard format.

My experiments outputs (which you might need) are in `<OLIVIERI_PRIVATE_PATH>/exps`.

### Cache Prefilling

Experiments with text require a cache checkpoint that is already built for my experiments, but might have to be built for different experiments.

To build cache for an experiment, prepare the config.yml (A) for that experiment, navigate the `src/cache` module and modify the config.yml (B) there:
	
- `checkpoint_config` should point to the original experiment config (A).
- `target_path`should point to the path in which the cache will be stored (I usually keep it in the same of the experiments outputs).

Then, `cd` to the `src` folder and execute the following command:
```bash
python3 -m cache.prefill_cache
```

The cache prefilling mechanism injects the same config of the original experiment, initialises the cache from scratch and performs an entire epoch with no weight updates to accumulate the masks.

Next, you go in the original experiment config (A) and point the cache checkpoint path to the one you just created.

## Experiments

1. **XEn Rescale**: `src/train/seg/2-xen-rescale`
2. **Contrastive B || dT**: `src/train/seg/3-contrastive-B_vs_dT`
	1. FT synthetic bootstrapping: `3.1-ft_synth_bootstrap`
	2. synthetic bootstrapping: *TBD*
	3. full training: *TBD*
3. **Contrastive B_PR || dT**: *TBD*

