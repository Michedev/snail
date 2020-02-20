# SNAIL (Simple Neural Attentive Meta-Learner)
Pytorch implementation of SNAIL network trained on Omniglot/Mini imagenet

## How to train the model

You can train the model via train.py

the script accepts many input arguments which are

    DESCRIPTION
        Download the dataset if not present and train SNAIL (Simple Neural Attentive Meta-Learner).
        When training is successfully finished, the embedding network weights and snail weights are saved under model_weights/,
        as well the path of classes used for training/test in train_classes.txt/test_classes.txt

    FLAGS
        --dataset=DATASET
            Dataset used for training,  can be only {'omniglot', 'miniimagenet'} (defuult 'omniglot')
        --n=N
            the N in N-way in meta-learning i.e. number of class sampled in each row of the dataset (default 5)
        --k=K
            the K in K-shot in meta-learning i.e. number of observations for each class (default 5)
        --trainsize=TRAINSIZE
            number of class used in training (default 1200) (the remaining classes are for test)
        --epochs=EPOCHS
            times that model see the dataset (default 200)
        --batch_size=BATCH_SIZE
            size of a training batch (default 32)
        --random_rotation=RANDOM_ROTATION
            :bool rotate the class images by multiples of 90 degrees (default True)
        --seed=SEED
            seed for reproducibility (default 13)
        --force_download=FORCE_DOWNLOAD
            :bool redownload data even if folder is present (default True)
        --device=DEVICE
            : device used in pytorch for training, can be "cuda*" or "cpu" (default 'cuda')
        --use_tensorboard=USE_TENSORBOARD
            :bool save metrics in tensorboard (default True)
        --eval_test=EVAL_TEST
            :bool after test_loss_freq batch calculate loss and accuracy on test set (default True)
        --track_loss_freq=TRACK_LOSS_FREQ
            :int step frequency of loss/accuracy save into tensorboard (default 10)
        --load_weights=LOAD_WEIGHTS
            :bool if available load under model_weights snail and embedding network weights (default True)
