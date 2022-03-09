# cfg 

class Config:
    augmentation='CCRAugmentation'
    batch_size=32
    criterion='cross_entropy'
    model_dir='./model'
    data_dir='/opt/ml/input/data/train/images'
    dataset='CustomDataset'
    epochs=10
    log_interval=20
    lr=0.0001 # 수정 
    lr_decay_step=20 
    model='MyEfficientNet'
    name='exp'
    optimizer='Adam' # sgd-lr크게
    resize=[128, 96]
    seed=42
    val_ratio=0.2
    valid_batch_size=100
    fold_num = 5
    drop_out = 0.7 ### 
    