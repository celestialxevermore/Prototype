# #!/bin/bash

# gpu_id=4
# baseline="xgb"
# #datasets="heart heart heart heart heart heart"
# datasets="heart"
# seeds="42 44 46 48 50"
# few_shots="4 8 16 32 64"
# base_dir="ML_test20260102"

# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64

# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Heart_disease_statlog --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64


# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Cardiovascular_Disease_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64


# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Medicaldataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64


# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data heart_failure_clinical_records --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64


# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data cardio_SAheart --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64


# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 42 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 44 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 46 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 48 --few_shot 64
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 4
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 8
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 16
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 32
# OMP_NUM_THREADS=10 python main_ml.py --baseline xgb --source_data Erbil_Cardiovascular_Health_Dataset --base_dir ML_${baseline}_20260102 --random_seed 50 --few_shot 64

#!/bin/bash

gpu_id=4
baseline="xgb"
datasets="heart"
seeds="42 44 46 48 50"
few_shots="4 8 16 32 64"
base_dir="ML_test20260105"   # ✅ 여기 공통으로 고정

for dataset in $datasets; do
  for shot in $few_shots; do
    for seed in $seeds; do
      echo "Running - dataset:$dataset, baseline:$baseline, few_shot:$shot, seed:$seed"
      CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_ml.py \
        --source_data $dataset \
        --few_shot $shot \
        --baseline $baseline \
        --random_seed $seed \
        --base_dir $base_dir \
        --des ML_${baseline}_20260105
    done
  done
done
