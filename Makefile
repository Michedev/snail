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