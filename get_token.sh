#!/bin/bash

docker exec -it jupyter jupyter notebook list | grep -o -E "token=([0-9a-z]*) " | sed -e "s|token=||g"
