# Intentionally invalid. Use build_image.py with a reviewed name@sha256 reference.
ARG TRAINER_BASE_IMAGE=REPLACE_WITH_REVIEWED_BASE_IMAGE_AT_SHA256_DIGEST
FROM ${TRAINER_BASE_IMAGE}

ARG SOURCE_REVISION
ARG SOURCE_TREE_SHA256
ARG VERL_REVISION=8fdc4d3f202f41461f4de9f42a637228e342668b
ARG TRAINER_BASE_NAME
ARG TRAINER_BASE_DIGEST

LABEL org.opencontainers.image.title="Compute as a Teacher trainer" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.base.name="${TRAINER_BASE_NAME}" \
      org.opencontainers.image.base.digest="${TRAINER_BASE_DIGEST}" \
      io.compute-as-a-teacher.verl.revision="${VERL_REVISION}"

USER root

RUN test "${VERL_REVISION}" = "8fdc4d3f202f41461f4de9f42a637228e342668b" \
    && mkdir -p /opt/verl/source \
    && git init /opt/verl/source \
    && git -C /opt/verl/source remote add origin https://github.com/volcengine/verl.git \
    && git -C /opt/verl/source fetch --depth=1 origin "${VERL_REVISION}" \
    && test "$(git -C /opt/verl/source rev-parse FETCH_HEAD)" = "${VERL_REVISION}" \
    && git -C /opt/verl/source checkout --detach "${VERL_REVISION}"

RUN groupadd --gid 10001 catrun \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash catrun \
    && mkdir -p /opt/cat/repo /opt/cat/image-metadata /mnt/models /mnt/data/math500 /mnt/outputs /mnt/config \
    && chown -R 10001:10001 /opt/cat /mnt/outputs

# The allowlist is deliberate: models, datasets, outputs, notebooks, the paper,
# credentials, and the Git database cannot enter the image from the build context.
COPY --chown=10001:10001 pyproject.toml uv.lock README.md /opt/cat/repo/
COPY --chown=10001:10001 src /opt/cat/repo/src
COPY --chown=10001:10001 scripts/train_math500.py scripts/evaluate_math500.py scripts/prepare_math500.py scripts/server_math500.py /opt/cat/repo/scripts/
COPY --chown=10001:10001 prompts /opt/cat/repo/prompts
COPY --chown=10001:10001 configs/datasets /opt/cat/repo/configs/datasets
COPY --chown=10001:10001 configs/evals /opt/cat/repo/configs/evals
COPY --chown=10001:10001 configs/training /opt/cat/repo/configs/training
COPY --chown=10001:10001 configs/server /opt/cat/repo/configs/server
COPY --chown=10001:10001 infra/server /opt/cat/repo/infra/server

RUN test -z "$(find /opt/cat/repo /opt/verl/source -type f \( \
        -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name '*.ckpt' \
        -o -name '*.pt' -o -name '*.pth' -o -name '*.gguf' \) -print -quit)" \
    && ln -s /mnt/data /opt/cat/repo/data \
    && ln -s /mnt/outputs /opt/cat/repo/outputs

RUN test "$(uname -m)" = "x86_64" \
    && python -m pip install --disable-pip-version-check --no-input --no-cache-dir \
        --force-reinstall --no-deps --require-hashes --only-binary=:all: \
        -r /opt/cat/repo/infra/server/overlay-requirements.lock \
    && python -m pip check

RUN python /opt/cat/repo/infra/server/write_image_metadata.py \
        --source-revision "${SOURCE_REVISION}" \
        --base-name "${TRAINER_BASE_NAME}" \
        --base-digest "${TRAINER_BASE_DIGEST}" \
        --verl-revision "${VERL_REVISION}" \
        --source-root /opt/cat/repo \
        --expected-source-tree-sha256 "${SOURCE_TREE_SHA256}" \
        --output /opt/cat/image-metadata/source.json

ENV HOME=/home/catrun \
    PYTHONPATH=/opt/cat/repo/src:/opt/verl/source \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=true \
    CAT_VERL_SOURCE=/opt/verl/source \
    CAT_MODEL_DIR=/mnt/models/required_full_model_commit_sha \
    CAT_DATA_DIR=/mnt/data \
    CAT_OUTPUT_DIR=/mnt/outputs \
    CAT_CONFIG_PATH=/mnt/config/training.toml \
    CAT_WORKFLOW_PATH=/mnt/config/workflow.toml \
    XDG_CACHE_HOME=/mnt/outputs/cache \
    TRITON_CACHE_DIR=/mnt/outputs/cache/triton \
    WANDB_DIR=/mnt/outputs/wandb \
    WANDB_CACHE_DIR=/mnt/outputs/wandb-cache

WORKDIR /opt/cat/repo
USER 10001:10001

# This also records the complete Python distribution inventory in the image.
RUN python infra/server/doctor.py --mode image \
    --inventory-out /opt/cat/image-metadata/python-packages.json

HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD ["python", "infra/server/doctor.py", "--mode", "health", "--quiet"]

ENTRYPOINT ["python", "infra/server/entrypoint.py"]
CMD ["ready"]
