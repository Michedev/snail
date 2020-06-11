clean-log-omniglot:
	rm -r tb/*omniglot*

clean-log-miniimagenet:
	rm -r tb/*miniimagenet*

clean-log-all: clean-log-miniimagenet clean-log-omniglot

clean-miniimagenet-model:
	rm model_weights/*miniimagenet*.pth

clean-omniglot-model:
	rm model_weights/*omniglot*.pth

clean-all-model: clean-miniimagenet-model clean-omniglot-model