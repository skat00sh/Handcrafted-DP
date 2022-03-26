for SIGMA in 0
do
	for CLIP_NORM in 0.1 1
	do
	  python3 cnns.py --dataset=mnist --batch_size=512 --lr=0.5 --epochs=3 --save_checkpoint_per_epoch=1 --noise_multiplier=$SIGMA --max_grad_norm=$CLIP_NORM --checkpoint_save_path=/opt/sperl/students/devendra/projects/Handcrafted-DP
	done

done