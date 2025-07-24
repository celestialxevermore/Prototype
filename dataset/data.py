# Utility function for getting data & prompting & query
import os
import random
import openai
import time
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

''' 
    FEATLLM get_dataset
    get_dataset : 전체 데이터셋을 받아서, shot의 개수만큼만 train_test_split
    ex. adult datset 
    (48842, 15)
    X_train : (39073, 14)
    X_test : (9769, 14)
    y_train : (39073,)
    y_test : (9769,)

    get_dataset result 
    X_train : (64, 14)
    y_train : (64,)

'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_dataset(args, data_name, seed):
    # args.table_path를 사용하여 파일 경로 설정
    file_name = os.path.join(args.table_path, f"origin_table/{data_name}.csv")

    df = pd.read_csv(file_name, index_col=0)
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator


def get_dataset_ml(args, data_name, seed):
    # args.table_path를 사용하여 파일 경로 설정
    file_name = os.path.join(args.table_path, f"origin_table/{data_name}.csv")

    df = pd.read_csv(file_name, index_col=0)
    
    # 특정 feature 제거 로직 추가
    if args.del_feat:
        available_features = df.columns.tolist()
        features_to_remove = []
        
        for feat in args.del_feat:
            if feat in available_features:
                features_to_remove.append(feat)
                print(f"Removing feature: {feat} from {data_name} dataset")
            else:
                print(f"Warning: Feature {feat} not found in {data_name} dataset. Available features: {available_features}")
        
        # 실제 feature 제거
        if features_to_remove:
            df = df.drop(columns=features_to_remove)
            print(f"Successfully removed {len(features_to_remove)} features: {features_to_remove}")
    
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator
