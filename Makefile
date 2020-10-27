clean-log-omniglot:
	rm -r tb/*omniglot*

clean-log-miniimagenet:
	rm -r tb/*miniimagenet*

clean-log-all: clean-log-miniimagenet clean-log-omniglot

clean-model-miniimagenet:
	rm model_weights/*miniimagenet*.pth

clean-model-omniglot:
	rm model_weights/*omniglot*.pth

clean-pretrained:
	rm model_weights/embedding-pretraining/*

clean-all-miniimagenet: clean-log-miniimagenet clean-model-miniimagenet

clean-all-omniglot: clean-log-omniglot clean-model-omniglot

clean-model-all: clean-miniimagenet-model clean-omniglot-model


BATCHSIZEPRETRAINING:=16
DEVICETRAIN := 'cuda::0'

pretrain-miniimagenet:
	python3 pretraining.py --epochs=5 --batch-size=$(BATCHSIZEPRETRAINING) --device=$(DEVICETRAIN)


train-miniimagenet-n5-k1:
	python3 train.py --dataset='miniimagenet' --n=5 --k=1 --epochs=100 --batch-size=8 --trainsize=10000 --testsize=64 --device=$(DEVICETRAIN) --evalength=64 --lr=10e-4 --track-weights=True --train-weights-freq=1000 --random-rotation=False --trainpbar=False

train-miniimagenet-n5-k1-no-pretrain:
	python3 train.py --dataset='miniimagenet' --n=5 --k=1 --epochs=100 --batch-size=8 --trainsize=10000 --testsize=64 --device=$(DEVICETRAIN) --evalength=64 --lr=10e-4 --track-weights=True --train-weights-freq=1000 --random-rotation=False --trainpbar=False --use-pretraining=False --init-truncated-normal

train-miniimagenet-n5-k1-with-logs:
	python3 train.py --dataset='miniimagenet' --n=5 --k=1 --epochs=100 --batch-size=8 --trainsize=10000 --testsize=64 --device=$(DEVICETRAIN) --evalength=64 --lr=10e-5 --track-weights=True --train-weights-freq=1000 --random-rotation=False --trainpbar=False > train-log.txt 2> train-err.txt



train-miniimagenet-n5-k5:
	python3 train.py --dataset='miniimagenet' --n=5 --k=5 --epochs=100 --batch-size=8 --trainsize=10000 --testsize=64 --device='cuda:1' --evalength=64 --lr=10e-5 --track-weights=True --train-weights-freq=1000 --random-rotation=False --trainpbar=False > train-log.txt 2> train-err.txt
	
	
