import json

def build_config(dataset, cv_train_views=None, cv_test_views=None, training_mode='CS'):
    cfg = type('', (), {})() 
    
    if dataset == 'ntu_rgbd_120':
        cfg.videos_folder =  '/media/abc/soft/ljq_dataset/rgb_hdf5/'
        if training_mode == 'CV':
            cfg.train_annotations = 'data/NTUTrain_CSetmap.csv'
            cfg.test_annotations = 'data/NTUTrain_CSetmap.csv'
        else:  # CSet and CS are modified here by default.
            cfg.train_annotations = 'data/NTUTrain_CSmap.csv'
            cfg.test_annotations = 'data/NTUTest_CSmap.csv'
        cfg.num_actions = 120
        # A second check is performed to ensure the viewing angle is correct.
        cfg.cv_train_views = cv_train_views if cv_train_views is not None else None
        cfg.cv_test_views = cv_test_views if cv_test_views is not None else None
        
    elif dataset == 'ntu_rgbd_60':
        cfg.videos_folder =  '/media/abc/ware/ljq/datasets/NTU/rgb_hdf5/'  # Points to the HDF5 file directory.
        # Select the annotation file based on the training mode.
        if training_mode == 'CV':
            cfg.train_annotations = 'data/NTU60Train_CVmap.csv'
            cfg.test_annotations = 'data/NTU60Test_CVmap.csv'
        else: 
            cfg.train_annotations = 'data/NTU60Train_CSmap.csv'
            cfg.test_annotations = 'data/NTU60Test_Csmap.csv'
        cfg.num_actions = 60
        # A second check is performed to ensure the viewing angle is correct.
        cfg.cv_train_views = cv_train_views if cv_train_views is not None else [2, 3]
        cfg.cv_test_views = cv_test_views if cv_test_views is not None else [1]

        
    elif dataset == "pkummd":
        cfg.videos_folder =  '/media/abc/soft/ljq_dataset/pku/pku-mmd-v1-video-hdf5/'
        # Select the annotation file based on the training mode.
        if training_mode == 'CV':
            cfg.train_annotations = 'data/PKUMMDTrainCV_map_MR.csv'
            cfg.test_annotations = 'data/PKUMMDTestCV_map_L.csv'
        else:
            cfg.train_annotations = 'data/PKUMMDTrainCS_map_cleaned.csv'
            cfg.test_annotations = 'data/PKUMMDTestCS_map_cleaned.csv'
        cfg.num_actions = 51
        # A second check is performed to ensure the viewing angle is correct.
        cfg.cv_train_views = cv_train_views if cv_train_views is not None else [2, 3]
        cfg.cv_test_views = cv_test_views if cv_test_views is not None else [1]

    elif dataset == 'numa':
        cfg.videos_folder =  '/media/abc/ware/ljq/datasets/numa/'
        cfg.num_actions = 10
        # Select the annotation file based on the training mode.
        if training_mode == 'CV':
            cfg.train_annotations = "data/NUMATrain_CV_cleaned.csv"
            cfg.test_annotations = "data/NUMATest_CV_cleaned.csv"
        else: 
            cfg.train_annotations = "data/NUMATrain_CS_cleaned.csv"
            cfg.test_annotations = "data/NUMATest_CS_cleaned.csv"
        # A second check is performed to ensure the viewing angle is correct.
        cfg.cv_train_views = cv_train_views if cv_train_views is not None else [1, 2]
        cfg.cv_test_views = cv_test_views if cv_test_views is not None else [3]
  
    else:
        raise NotImplementedError
        
    cfg.dataset = dataset
    cfg.saved_models_dir = './results/saved_models'
    cfg.outputs_folder = './results/outputs'
    cfg.tf_logs_dir = './results/logs'
    
    print(f"training_mode: {training_mode}")

    return cfg
