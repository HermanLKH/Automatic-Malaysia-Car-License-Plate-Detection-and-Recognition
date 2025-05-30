dataset_type = 'OCRDataset'
data_root = 'data/dataset_malaysia'

# Training
my_dataset_train = dict(
    type=dataset_type,
    data_prefix=dict(img_path=f'{data_root}/train_images/'),
    ann_file=f'{data_root}/train.json',
    metainfo={}
)

# Validation 
my_dataset_val = dict(
    type=dataset_type,
    data_prefix=dict(img_path=f'{data_root}/val_images/'),
    ann_file=f'{data_root}/val_test_merged.json',
    metainfo={}
)

my_dataset_test = my_dataset_val
