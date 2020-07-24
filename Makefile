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


epochs = 100
batchsize = 16
trainsize = 30000
lr = 10e-5
device = cuda:1
testsize = 64
trackweightssteps = 10000

train-miniimagenet-n5-k1:
	python3 train.py --dataset='miniimagenet' --n=5 --k=1 --epochs=$(epochs) --batch-size=$(batchsize) --trainsize=$(trainsize) --testsize=$(testsize) --device='$(device)' --evalength=$(testsize) --lr=$(lr) --track-weights=True --train-weights-freq=$(trackweightssteps) --random-rotation=False --trainpbar=True


train-miniimagenet-n5-k1-with-log:
	python3 train.py --dataset='miniimagenet' --n=5 --k=1 --epochs=$(epochs) --batch-size=$(batchsize) --trainsize=$(trainsize) --testsize=$(testsize) --device='$(device)' --evalength=$(testsize) --lr=$(lr) --track-weights=True --train-weights-freq=$(trackweightssteps) --random-rotation=False --trainpbar=True > train-log.txt 2> train-err.txt

batchsize = 8

train-miniimagenet-n5-k5:
	python3 train.py --dataset='miniimagenet' --n=5 --k=5 --epochs=$(epochs) --batch-size=$(batchsize) --trainsize=$(trainsize) --testsize=$(testsize) --device='$(device)' --evalength=$(testsize) --lr=$(lr) --track-weights=True --train-weights-freq=$(trackweightssteps) --random-rotation=False --trainpbar=True
