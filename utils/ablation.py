import torch
import numpy as np
import random
import copy
import os
import json
import logging
from typing import List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_important_features(model, data_loader, device, top_k=3):
    """
    학습된 모델에서 CLS 토큰의 attention weight를 분석하여 중요한 변수들을 추출
    
    Args:
        model: 학습된 모델
        data_loader: 데이터 로더
        device: 디바이스
        top_k: 상위 몇 개의 변수를 추출할지
    
    Returns:
        important_features: 중요한 변수들의 이름 리스트
        all_features: 전체 변수들의 이름 리스트
    """
    return model.extract_cls_attention_weights(data_loader, device, top_k)

def determine_del_feat_by_scenario(scenario_id, important_features, all_features, unimportant_features):
   """
   시나리오별로 제거할 변수 리스트 결정
   
   Args:
       scenario_id: 시나리오 번호 (1, 2, 3, 4)
       important_features: 중요한 변수들 리스트 (Top-3, attention 높은 순)
       all_features: 전체 변수들 리스트
       unimportant_features: 안중요한 변수들 리스트 (Bottom-3, attention 낮은 순)
   
   Returns:
       제거할 변수들의 리스트
   """
   if scenario_id == 1:
   # "중요한 3개 제거"
      return list(important_features)
   elif scenario_id == 2:
    # "중요한 3개 + 가장 안중요한 1개 제거"
      to_remove = list(important_features)  # 중요한 3개
      if len(unimportant_features) >= 1:
          most_unimportant = unimportant_features[0]  # 가장 안중요한 1개 (이미 정렬되어 있음)
          to_remove.append(most_unimportant)
      return to_remove
   elif scenario_id == 3:
    # "가장 안중요한 3개 제거" (attention이 가장 낮은 순)
        return list(unimportant_features) if len(unimportant_features) >= 3 else list(unimportant_features)
   elif scenario_id == 4:
    # "중요한 3개만 유지" = 나머지 모두 제거
        important_set = set(important_features)
        return [f for f in all_features if f not in important_set]
   else:
        return []

def save_scenario_results_individually(args, all_scenario_results, important_features, all_features, unimportant_features):
   from utils.util import prepare_results_, save_results_A
   
   for scenario_id, scenario_data in all_scenario_results.items():
       logger.info(f"Saving results for Scenario {scenario_id}...")
       
       args_scenario = copy.deepcopy(args)
       args_scenario.del_feat = scenario_data['removed_features']
       
       # prepare_results_가 기대하는 전체 구조로 맞춤
       full_ours_results = scenario_data['full_results']
       few_ours_results = scenario_data['few_results']
       
       # 기존 prepare_results_ 함수 사용
       combined_results = prepare_results_(full_ours_results, few_ours_results)
       combined_results['scenario_info'] = {
           'scenario_id': scenario_id,
           'scenario_description': scenario_data['description'],
           'important_features': important_features,
           'all_features': all_features,
           'removed_features': scenario_data['removed_features']
       }
       
       save_results_A(args_scenario, combined_results)
       logger.info(f"Scenario {scenario_id} results saved successfully")