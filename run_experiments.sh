for SIGMA in 0 1 1.5 2 3
do
        for CLIP_NORM in 0.1 1 3 5 10
        do
          python3 cnns.py --dataset=mnist --batch_size=512 --lr=0.5 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --checkpoint_save_path=/opt/sperl/students/devendra/projects/Handcrafted-DP --logdir=/opt/sperl/students/devendra/projects/Handcrafted-DP/logs
          python3 cnns.py --dataset=fmnist --batch_size=2048 --lr=4 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --checkpoint_save_path=/opt/sperl/students/devendra/projects/Handcrafted-DP --logdir=/opt/sperl/students/devendra/projects/Handcrafted-DP/logs
          python3 cnns.py --dataset=cifar10 --batch_size=1024 --lr=1 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --checkpoint_save_path=/opt/sperl/students/devendra/projects/Handcrafted-DP --logdir=/opt/sperl/students/devendra/projects/Handcrafted-DP/logs
        done

done
~