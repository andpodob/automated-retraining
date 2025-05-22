#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <experiment_name_prefix>"
  echo "Example: $0 my_custom_experiment"
  exit 1
fi

EXPERIMENT_PREFIX="$1"

for i in {0..4}; do
    COMMAND="python examples/gan/trainer/mass_trainer.py --exp_name ${EXPERIMENT_PREFIX}_${i}"
    echo "Executing: ${COMMAND}"
    ${COMMAND}
    echo "Command for synthetic_${i} finished."
done

echo "All mass_trainer.py processes for experiment ${EXPERIMENT_PREFIX} successfully."
