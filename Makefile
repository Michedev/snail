clean-log-omniglot:
	rm -r tb/*omniglot*

clean-log-miniimagenet:
	rm -r tb/*miniimagenet*

clean-log-all: clean-log-miniimagenet clean-log-omniglot

clean-model-miniimagenet:
	rm model_weights/*miniimagenet*.pth

clean-model-omniglot:
	rm model_weights/*omniglot*.pth

clean-all-miniimagenet: clean-log-miniimagenet clean-model-miniimagenet

clean-all-omniglot: clean-log-omniglot clean-model-omniglot

clean-model-all: clean-miniimagenet-model clean-omniglot-model


n = 5
k = 1
epochs = 100
batchsize = 16
trainsize = 30000
lr = 10e-4

train-miniimagenet-n5-k1:
	python3 train.py --dataset='miniimagenet' --n=$n --k=$k --epochs=$(epochs) --batch-size=$(batchsize) --trainsize=$(trainsize) --testsize=64 --device='cuda:1' --evalength=64 --lr=$(lr) --track-weights=True --train-weights-freq=10000 --random-rotation=False --trainpbar=True
	
	
