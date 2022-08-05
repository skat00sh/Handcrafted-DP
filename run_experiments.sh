for SIGMA in 0.1 0.2 0.5 1.0 1.5
do
        for CLIP_NORM in 0.1 0.2 0.3 0.5 1.0
        do
          python3 cnns.py --dataset=mnist --batch_size=512 --lr=0.5 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --results_path=/opt/sperl/students/devendra/projects/Handcrafted-DP/mulit_run_results/1 --logdir=/logs/mulit_run_results/1
          python3 cnns.py --dataset=fmnist --batch_size=2048 --lr=4 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --results_path=/opt/sperl/students/devendra/projects/Handcrafted-DP/mulit_run_results/1 --logdir=/logs/mulit_run_results/1
#          python3 cnns.py --dataset=cifar10 --batch_size=1024 --lr=1 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --checkpoint_save_path=/opt/sperl/students/devendra/projects/Handcrafted-DP --logdir=/opt/sperl/students/devendra/projects/Handcrafted-DP/logs
        done

done

## For CIFAR10
# for SIGMA in 0.1 0.2 0.5 1.0 1.5
# do
#         for CLIP_NORM in 0.1 0.2 0.3 0.5 
#         do
#          python3 cnns.py --dataset=cifar10 --batch_size=1024 --lr=1 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --checkpoint_save_path=/opt/sperl/students/devendra/projects/Handcrafted-DP/mulit_run_results/1 --logdir=/logs
#         done

# done

#####
## Run config for Docker
#####

# for SIGMA in 1.0 1.5
#   do
#     for CLIP_NORM in 0.3 0.5 1.0
#       do
#         python3 cnns.py --dataset=mnist --batch_size=512 --early_stop --lr=0.5 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --results_path=/workspace --logdir=/logs
#       done
#   done

# python3 cnns.py --dataset=mnist --batch_size=512 --early_stop --lr=0.5 --noise_multiplier=0.5 --max_grad_norm=1.0 --results_path=/workspace --logdir=/logs
