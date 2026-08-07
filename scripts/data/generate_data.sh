#!/bin/bash
set -e

DIR_PATH="./runtime/data"

if [ "$#" -eq 0 ]; then
    SEEDS=(42)
else
    SEEDS=("$@")
fi

echo "Generating data with seeds: ${SEEDS[*]}"
echo "Output dir: $DIR_PATH"

nntp --entrypoint=./src/entrypoint.py generate --task fdgo-ry reactgo-ry delaygo-ry --seed "${SEEDS[@]}"  --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path="$DIR_PATH"
nntp --entrypoint=./src/entrypoint.py generate --task fdanti-ry reactanti-ry delayanti-ry --seed "${SEEDS[@]}"  --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path="$DIR_PATH"
nntp --entrypoint=./src/entrypoint.py generate --task dm1-ry dm2-ry contextdm1-ry contextdm2-ry multidm-ry --seed "${SEEDS[@]}"  --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path="$DIR_PATH"
nntp --entrypoint=./src/entrypoint.py generate --task delaydm1-ry delaydm2-ry contextdelaydm1-ry contextdelaydm2-ry multidelaydm-ry --seed "${SEEDS[@]}"  --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path="$DIR_PATH"
# nntp --entrypoint=./src/entrypoint.py generate --task dmsgo-ry dmsnogo-ry dmcgo-ry dmcnogo-ry --seed "${SEEDS[@]}"  --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path="$DIR_PATH"

nntp --entrypoint=./src/entrypoint.py generate --task fdgo-ry reactgo-ry delaygo-ry --seed "${SEEDS[@]}"  --training-size 128000 --validation-size 4096 --test-size 4096 --dir-path="$DIR_PATH"
nntp --entrypoint=./src/entrypoint.py generate --task fdanti-ry reactanti-ry delayanti-ry --seed "${SEEDS[@]}"  --training-size 128000 --validation-size 4096 --test-size 4096 --dir-path="$DIR_PATH"
nntp --entrypoint=./src/entrypoint.py generate --task dm1-ry dm2-ry contextdm1-ry contextdm2-ry multidm-ry --seed "${SEEDS[@]}"  --training-size 128000 --validation-size 4096 --test-size 4096 --dir-path="$DIR_PATH"
nntp --entrypoint=./src/entrypoint.py generate --task delaydm1-ry delaydm2-ry contextdelaydm1-ry contextdelaydm2-ry multidelaydm-ry --seed "${SEEDS[@]}"  --training-size 128000 --validation-size 4096 --test-size 4096 --dir-path="$DIR_PATH"
# nntp --entrypoint=./src/entrypoint.py generate --task dmsgo-ry dmsnogo-ry dmcgo-ry dmcnogo-ry --seed "${SEEDS[@]}"  --training-size 128000 --validation-size 4096 --test-size 4096 --dir-path="$DIR_PATH"