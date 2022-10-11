#!/bin/bash
# setting training config here
batch_size=8
epochs=50
lr=0.0001
step=50
gamma=0.9
img_size=224
save_dir="checkpoint/vgg16_fcn8s"
checkpoint=""

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --step ${step} --gamma ${gamma} --img_size ${img_size} --save_dir ${save_dir}"
#config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 improved.py ${config}"

echo "${run}"
${run}
