# SNAIL (Simple Neural Attentive Meta-Learner)
Pytorch implementation of SNAIL network trained on Omniglot/Mini imagenet

## How to train the model

You can train the model via train.py

the script accepts many input arguments which are

    NAME
        train.py - Download the dataset if not present and train SNAIL (Simple Neural Attentive Meta-Learner). When training is successfully finished, the embedding network weights and snail weights are saved and the path of classes used for training/test in train_classes.txt/test_classes.txt

    SYNOPSIS
        train.py <flags>

    DESCRIPTION
        Download the dataset if not present and train SNAIL (Simple Neural Attentive Meta-Learner). When training is successfully finished, the embedding network weights and snail weights are saved and the path of classes used for training/test in train_classes.txt/test_classes.txt

    FLAGS
        --dataset=DATASET
            Dataset used for training,  can be only {'omniglot', 'miniimagenet'}
        --n=N
            the N in N-way in meta-learning i.e. number of class sampled in each row of the dataset
        --k=K
            the K in K-shot in meta-learning i.e. number of observations for each class
        --trainsize=TRAINSIZE
            number of class used in training
        --episodes=EPISODES
            time of model updates
        --batch_size=BATCH_SIZE
            size of a training batch
        --seed=SEED
            seed for reproducibility
        --force_download=FORCE_DOWNLOAD
            :bool redownload data even if folder is present
        --device=DEVICE
            : device used in pytorch for training, can be "cuda*" or "cpu"
        --use_tensorboard=USE_TENSORBOARD
            :bool save metrics in tensorboard
        --save_destination=SAVE_DESTINATION
            :string location of model weights
