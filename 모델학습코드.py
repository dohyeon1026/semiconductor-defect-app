import pandas as pd
import xgboost as xgb
import joblib

# 1. 입력 변수와 타겟 정의
input_cols = [
    'thickness', 'speed', 'coolant', 'LAP_pressure', 'fab_temp', 'fab_humidity',
    'blade_thickness', 'feed_rate', 'blade_speed', 'coolant_flow', 'SAW_fab_temp', 'SAW_fab_humidity',
    'Die_temp', 'Die_pressure', 'Die_time', 'viscosity', 'DIE_fab_temp', 'DIE_fab_humid',
    'wire_diameter', 'bond_force_1', 'bond_ultra_1', 'bond_time_1',
    'bond_force_2', 'bond_ultra_2', 'bond_time_2', 'ultra_freq',
    'Wire_fab_temp', 'Wire_fab_humidity', 'Mold_temp', 'Mold_pressure',
    'Mold_time', 'resin_viscosity', 'Mold_fab_temp', 'Mold_fab_humidity',
    'mark_laser_power', 'mark_pulse_freq', 'mark_speed', 'mark_depth',
    'mark_fab_temp', 'mark_fab_humidity'
]

target_cols = [
    "backlap_defect", "sawing_defect", "dieattach_defect",
    "wirebond_defect", "molding_defect", "marking_defect"
]

# 2. 데이터 로드
df = pd.read_csv("가상_공정_데이터.csv")

# 3. 모델 학습 및 저장
for target in target_cols:
    X_train = df[input_cols]
    y_train = df[target]

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 4. 모델 저장
    joblib.dump(model, f"{target}_model.pkl")
    print(f"? {target}_model.pkl 저장 완료")
