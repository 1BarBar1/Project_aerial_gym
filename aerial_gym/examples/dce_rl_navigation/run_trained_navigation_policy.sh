#!/usr/bin/env bash

python3 dce_nn_navigation.py --train_dir=$(pwd)/selected_network --experiment=selected_network --env=slope_forest_env --obs_key="observations" --load_checkpoint_kind=best
