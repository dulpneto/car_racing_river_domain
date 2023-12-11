#!/bin/bash

if [ "$LAMB" == "" ]; then
    echo "You must define LAMB"
    exit 1
fi

python3 train_model.py -l $LAMB > ../out.log