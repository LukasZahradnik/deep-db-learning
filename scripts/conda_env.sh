#!/bin/bash

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$1"

# Use this to suppress error codes that may interrupt the job or interfere with reporting.
# Do show the user that an error code has been received and is being suppressed.
# see https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command 
eval "${@:2}" || echo "Exit with error code $? (suppressed)"
