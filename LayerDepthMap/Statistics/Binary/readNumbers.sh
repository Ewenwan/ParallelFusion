#!/bin/bash

cat output | awk '{ print $5 " " $8 }' > output_values
