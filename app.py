import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Windows에서는 'malgun.ttf' (맑은 고딕)를 사용
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
else:
    font_path = os.path.join("fonts", "malgun.ttf")  # 프로젝트에 복사한 폰트

if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
else:
    print("?? 폰트 파일을 찾을 수 없습니다:", font_path)

# 마이너스 기호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")

# 이미지 안전 표시 함수 (Streamlit 버전에 따라 대응)
def safe_image(path, caption="", **kwargs):
    try:
        st.image(path, caption=caption, use_container_width=True, **kwargs)
    except TypeError:
        st.image(path, caption=caption, **kwargs)

# 1. 공정 변수 리스트
input_cols = [
    'thickness', 'speed', 'coolant', 'LAP_pressure', 'fab_temp', 'fab_humidity',
    'blade_thickness', 'feed_rate', 'blade_speed', 'coolant_flow', 'SAW_fab_temp', 'SAW_fab_humidity',
    'Die_temp', 'Die_pressure', 'Die_time', 'viscosity', 'DIE_fab_temp', 'DIE_fab_humid',
    'wire_diameter', 'bond_force_1', 'bond_ultra_1', 'bond_time_1',
    'bond_force_2', 'bond_ultra_2', 'bond_time_2', 'ultra_freq',
    'Wire_fab_temp', 'Wire_fab_humidity','Mold_temp', 'Mold_pressure', 'Mold_time',
    'resin_viscosity', 'Mold_fab_temp', 'Mold_fab_humidity',
    'mark_laser_power', 'mark_pulse_freq', 'mark_speed', 'mark_depth',
    'mark_fab_temp', 'mark_fab_humidity'
]

# 2. 변수 입력 범위
range_dict = {
    'thickness': (150, 450), 'speed': (25, 100), 'coolant': (5, 20), 'LAP_pressure': (10, 50),
    'fab_temp': (20, 24), 'fab_humidity': (35, 45), 'blade_thickness': (15, 60), 'feed_rate': (7, 15),
    'blade_speed': (30, 110), 'coolant_flow': (8, 20), 'SAW_fab_temp': (20, 24), 'SAW_fab_humidity': (35, 45),
    'Die_temp': (150, 200), 'Die_pressure': (10, 50), 'Die_time': (10, 60), 'viscosity': (1000, 4000),
    'DIE_fab_temp': (20, 24), 'DIE_fab_humid': (35, 45), 'wire_diameter': (203, 381),
    'bond_force_1': (35, 50), 'bond_ultra_1': (35, 50), 'bond_time_1': (10, 20),
    'bond_force_2': (90, 130), 'bond_ultra_2': (90, 130), 'bond_time_2': (10, 20),
    'ultra_freq': (100, 150), 'Wire_fab_temp': (20, 24), 'Wire_fab_humidity': (35, 45),
    'Mold_temp': (150, 200), 'Mold_pressure': (80, 140), 'Mold_time': (200, 300),
    'resin_viscosity': (1e7, 1e8), 'Mold_fab_temp': (20, 24), 'Mold_fab_humidity': (35, 45),
    'mark_laser_power': (13, 20), 'mark_pulse_freq': (10, 50), 'mark_speed': (67, 200),
    'mark_depth': (16, 72), 'mark_fab_temp': (20, 24), 'mark_fab_humidity': (35, 45),
}

# 3. 타겟 컬럼
target_cols = [
    "backlap_defect", "sawing_defect", "dieattach_defect",
    "wirebond_defect", "molding_defect", "marking_defect"
]

# 4. 세부 불량률
detail_cols = [
    'backlap_thickness_defect','backlap_speed_defect','backlap_LAP_pressure_defect','backlap_coolant_defect',
    'backlap_temp_defect','backlap_humidity_defect','sawing_blade_thickness_defect','sawing_feed_rate_defect',
    'sawing_blade_speed_defect','sawing_coolant_flow_defect','sawing_fab_humidity_defect','sawing_fab_temp_defect',
    'dieattach_temp_defect','dieattach_pressure_defect','dieattach_time_defect','dieattach_viscosity_defect',
    'dieattach_fab_temp_defect','dieattach_fab_humid_defect','wirebond_wire_diameter_defect',
    'wirebond_bond_force_1_defect','wirebond_bond_time_1_defect','wirebond_bond_ultra_1_defect',
    'wirebond_bond_force_2_defect','wirebond_bond_time_2_defect','wirebond_bond_ultra_2_defect',
    'wirebond_ultra_freq_defect','wirebond_fab_temp_defect','wirebond_fab_humidity_defect',
    'molding_Mold_temp_defect','molding_Mold_pressure_defect','molding_Mold_time_defect','molding_resin_viscosity_defect',
    'molding_Mold_fab_temp_defect','molding_Mold_fab_humidity_defect','marking_laser_power_defect',
    'marking_pulse_freq_defect','marking_speed_defect','marking_depth_defect','marking_fab_temp_defect',
    'marking_fab_humidity_defect'
]

# 공정변수별 불량률 매핑
variable_to_target = {
    # Backlap 공정
    'thickness': ['backlap_defect'],
    'speed': ['backlap_defect'],
    'coolant': ['backlap_defect'],
    'LAP_pressure': ['backlap_defect'],
    'fab_temp': ['backlap_defect'],
    'fab_humidity': ['backlap_defect'],

    # Sawing 공정
    'blade_thickness': ['sawing_defect'],
    'feed_rate': ['sawing_defect'],
    'blade_speed': ['sawing_defect'],
    'coolant_flow': ['sawing_defect'],
    'SAW_fab_temp': ['sawing_defect'],
    'SAW_fab_humidity': ['sawing_defect'],

    # Die Attach
    'Die_temp': ['dieattach_defect'],
    'Die_pressure': ['dieattach_defect'],
    'Die_time': ['dieattach_defect'],
    'viscosity': ['dieattach_defect'],
    'DIE_fab_temp': ['dieattach_defect'],
    'DIE_fab_humid': ['dieattach_defect'],

    # Wirebond
    'wire_diameter': ['wirebond_defect'],
    'bond_force_1': ['wirebond_defect'],
    'bond_ultra_1': ['wirebond_defect'],
    'bond_time_1': ['wirebond_defect'],
    'bond_force_2': ['wirebond_defect'],
    'bond_ultra_2': ['wirebond_defect'],
    'bond_time_2': ['wirebond_defect'],
    'ultra_freq': ['wirebond_defect'],
    'Wire_fab_temp': ['wirebond_defect'],
    'Wire_fab_humidity': ['wirebond_defect'],

    # Molding
    'Mold_temp': ['molding_defect'],
    'Mold_pressure': ['molding_defect'],
    'Mold_time': ['molding_defect'],
    'resin_viscosity': ['molding_defect'],
    'Mold_fab_temp': ['molding_defect'],
    'Mold_fab_humidity': ['molding_defect'],

    # Marking
    'mark_laser_power': ['marking_defect'],
    'mark_pulse_freq': ['marking_defect'],
    'mark_speed': ['marking_defect'],
    'mark_depth': ['marking_defect'],
    'mark_fab_temp': ['marking_defect'],
    'mark_fab_humidity': ['marking_defect']
}

# --- 모델 불러오기 ---
import os

@st.cache_resource
def load_models():
    models = {}
    base_path = os.path.dirname(os.path.abspath(__file__))  # app.py의 절대 경로
    for col in target_cols:
        model_path = os.path.join(base_path, "model", f"{col}_model.pkl")
        models[col] = joblib.load(model_path)
    return models

# --- 예측 함수 ---
def predict_all(input_data, full_df, models):
    row = pd.DataFrame([input_data], columns=input_cols)
    result = {}
    for col in target_cols:
        result[col] = round(models[col].predict(row)[0], 6)
    normal_probs = [1 - result[t] for t in target_cols]
    result['final_defect'] = round(1 - np.prod(normal_probs), 6)
    dist = ((full_df[input_cols] - row.iloc[0])**2).sum(axis=1)
    closest_row = full_df.iloc[dist.idxmin()]
    for d_col in detail_cols:
        result[d_col] = round(closest_row[d_col], 6)
    return result

# --- 조정 제안 함수 (변경: 중요도 → 실제 영향 기준) ---
def suggest_adjustments(models, user_input):
    suggestions = {}
    for col in target_cols:
        model = models[col]
        if hasattr(model, 'feature_importances_'):
            min_val = float('inf')
            best_val = None
            most_impact_var = None
            current_vals = dict(zip(input_cols, user_input))

            # 각 변수에 대해 현재값 유지 + 하나씩 바꿔가며 영향 평가
            impacts = {}
            for i, var in enumerate(input_cols):
                vals = np.linspace(range_dict[var][0], range_dict[var][1], 20)
                preds = []
                for v in vals:
                    temp_input = current_vals.copy()
                    temp_input[var] = v
                    temp_input = apply_all_correlations(temp_input)
                    row = pd.DataFrame([temp_input[col] for col in input_cols]).T
                    row.columns = input_cols
                    pred = model.predict(row)[0]
                    preds.append(pred)
                impact = max(preds) - min(preds)
                impacts[var] = impact

            # 가장 영향이 큰 변수 찾기
            most_impact_var = max(impacts, key=impacts.get)
            current_val = current_vals[most_impact_var]
            min_v, max_v = range_dict[most_impact_var]

            # 최적값 탐색
            scan_vals = np.linspace(min_v, max_v, 50)
            min_defect = float('inf')
            optimal_val = current_val
            for v in scan_vals:
                temp_input = current_vals.copy()
                temp_input[most_impact_var] = v
                temp_input = apply_all_correlations(temp_input)
                row = pd.DataFrame([temp_input[col] for col in input_cols]).T
                row.columns = input_cols
                pred = model.predict(row)[0]
                if pred < min_defect:
                    min_defect = pred
                    optimal_val = v

            suggestions[col] = {
                "variable": most_impact_var,
                "current": current_val,
                "optimal": optimal_val,
                "suggestion": f"'{most_impact_var}' 값을 {optimal_val:.2f}로 설정하면 불량률을 최소화할 수 있습니다."
            }
    return suggestions

def apply_correlation(variable_name, value, base_input):
    updated = base_input.copy()
    updated[variable_name] = value

    if variable_name == 'speed':
        updated['coolant'] += 0.15 * (value - 60)
        updated['thickness'] -= 0.01 * (value - 60)

    elif variable_name == 'LAP_pressure':
        updated['thickness'] -= 0.03 * (value - 30)

    elif variable_name == 'fab_temp':
        updated['thickness'] += 0.36 * (value - 22)

    elif variable_name == 'fab_humidity':
        updated['coolant'] += 0.2 * (value - 40)

    elif variable_name == 'blade_thickness':
        updated['blade_speed'] -= 0.05 * (value - 0.3)

    elif variable_name == 'SAW_fab_temp':
        updated['blade_thickness'] -= 0.01 * (value - 22)

    elif variable_name == 'SAW_fab_humidity':
        updated['feed_rate'] -= 0.1 * (value - 40)  

    elif variable_name == 'Die_temp':
        updated['viscosity'] -= 20 * (value - 175)
        updated['Die_pressure'] -= 0.5 * (value - 175)
        updated['Die_time'] -= 0.5 * (value - 175)

    elif variable_name == 'Die_pressure':
        updated['Die_time'] -= 0.05 * (value - 25)

    elif variable_name == 'viscosity':
        updated['Die_pressure'] += 0.005 * (value - 2500)
        updated['Die_time'] += 0.001 * (value - 2500)

    elif variable_name == 'DIE_fab_humid':
        updated['viscosity'] += 50 * (value - 40)

    elif variable_name == 'DIE_fab_temp':
        updated['Die_time'] += 1 * (22 - value)

    elif variable_name == 'Wire_fab_temp':
        updated['wire_diameter'] += 0.36 * (value - 22)


    elif variable_name in ['bond_force_1', 'bond_force_2']:
        total_force = (
            updated['bond_force_1'] if variable_name == 'bond_force_2' else value
        ) + (
            updated['bond_force_2'] if variable_name == 'bond_force_1' else value
        )
        updated['ultra_freq'] -= 0.02 * (total_force - 125)

    elif variable_name == 'Mold_fab_humidity':
        updated['resin_viscosity'] += 5e5 * (value - 40)

    elif variable_name == 'Mold_temp':
        updated['resin_viscosity'] -= 2e5 * (value - 175)
        updated['Mold_time'] -= 2 * (value - 175)

    elif variable_name == 'Mold_fab_temp':
        updated['Mold_time'] += 10 * (22 - value)

    elif variable_name == 'Mold_pressure':
        updated['resin_viscosity'] += 1e5 * (value - 128) / 48  # 반대 방향 정의

    elif variable_name == 'Mold_time':
        updated['resin_viscosity'] += 1e5 * (value - 250) / 100  

    elif variable_name == 'mark_laser_power':
        updated['mark_pulse_freq'] += 0.2 * (value - 16.5)
        updated['mark_speed'] += 1.5 * (value - 16.5)
        updated['mark_depth'] += 0.8 * (value - 16.5)

    elif variable_name == 'mark_pulse_freq':
        updated['mark_depth'] -= 0.4 * (value - 30)
        updated['mark_speed'] += 0.3 * (value - 30)

    elif variable_name == 'mark_speed':
        updated['mark_depth'] -= 0.3 * (value - 130)

    elif variable_name == 'mark_fab_temp':
        delta_depth = 0.01 * (value - 20)
        variation = np.random.normal(0, delta_depth)
        updated['mark_depth'] += variation

    return updated

def apply_all_correlations(base_input, max_iter=20, tol=1e-12):
    prev = base_input.copy()
    for _ in range(max_iter):
        updated = prev.copy()
        for var in prev:
            updated = apply_correlation(var, updated[var], updated)

        # 수렴 확인 (모든 값이 거의 변화 없으면 종료)
        diff = sum(abs(updated[k] - prev[k]) for k in updated if isinstance(updated[k], (int, float)))
        if diff < tol:
            break
        prev = updated
    return updated




# 3. 매핑 함수
def get_related_targets(variable):
    return variable_to_target.get(variable, target_cols)
                                  
def plot_defect_trend(variable_name, user_input, df, models):
    base_input = dict(zip(input_cols, user_input))
    values = np.linspace(range_dict[variable_name][0], range_dict[variable_name][1], 100)

    # 1) 공정별 불량률 예측
    related_targets = get_related_targets(variable_name)
    proc_preds = {target: [] for target in related_targets}
    for val in values:
        temp_input = base_input.copy()
        temp_input[variable_name] = val
        row = pd.DataFrame([temp_input], columns=input_cols)
        for target in related_targets:
            proc_pred = models[target].predict(row)[0]
            proc_preds[target].append(proc_pred * 100)

    # 2) 전체 불량률 예측
    total_preds = []
    for val in values:
        temp_input = base_input.copy()
        temp_input[variable_name] = val
        row = pd.DataFrame([temp_input], columns=input_cols)
        pred_total = 1 - np.prod([1 - models[col].predict(row)[0] for col in target_cols])
        total_preds.append(pred_total * 100)

    # 3) 상관관계 반영 전체 불량률 예측
    total_preds_corr = []
    for val in values:
        corr_input = apply_correlation(variable_name, val, base_input)
        row_corr = pd.DataFrame([corr_input], columns=input_cols)
        pred_corr = 1 - np.prod([1 - models[col].predict(row_corr)[0] for col in target_cols])
        total_preds_corr.append(pred_corr * 100)

    # 그래프1: 공정별 불량률
    fig_proc, ax_proc = plt.subplots(figsize=(8, 4))
    for target in related_targets:
        ax_proc.plot(values, proc_preds[target], label=f"{target} 불량률 (%)")
    idx = np.abs(values - base_input[variable_name]).argmin()
    for target in related_targets:
        ax_proc.scatter([base_input[variable_name]], [proc_preds[target][idx]], s=50)
    ax_proc.set_xlabel(variable_name)
    ax_proc.set_ylabel("공정별 불량률 (%)")
    ax_proc.set_title(f"'{variable_name}' 변화에 따른 공정별 불량률 예측")
    ax_proc.legend()
    ax_proc.grid(True)

    # 그래프2: 전체 불량률
    fig_total, ax_total = plt.subplots(figsize=(8, 4))
    ax_total.plot(values, total_preds, label="전체 불량률 (%)", color='tab:red')
    ax_total.scatter([base_input[variable_name]], [total_preds[idx]], color='red', s=50)
    ax_total.set_xlabel(variable_name)
    ax_total.set_ylabel("전체 불량률 (%)")
    ax_total.set_title(f"'{variable_name}' 변화에 따른 전체 불량률 예측")
    ax_total.legend()
    ax_total.grid(True)

    # 그래프3: 공정 변수 → 공정별 불량률 (기본 vs 상관관계 반영)
    fig_corr_proc, ax_corr_proc = plt.subplots(figsize=(8, 4))
    for target in related_targets:
        base_preds = proc_preds[target]
        corr_preds = []
        for val in values:
            temp_input = base_input.copy()
            temp_input[variable_name] = val
            corr_input = apply_all_correlations(temp_input)
            row_corr = pd.DataFrame([corr_input], columns=input_cols)
            pred_corr = models[target].predict(row_corr)[0]
            corr_preds.append(pred_corr * 100)

        ax_corr_proc.plot(values, base_preds, label=f"{target} (기본)", linestyle='--', color='gray')
        ax_corr_proc.plot(values, corr_preds, label=f"{target} (상관관계 반영)")
        ax_corr_proc.scatter([base_input[variable_name]], [base_preds[idx]], s=50, color='gray')
        ax_corr_proc.scatter([base_input[variable_name]], [corr_preds[idx]], s=50)

    ax_corr_proc.set_xlabel(variable_name)
    ax_corr_proc.set_ylabel("공정별 불량률 (%)")
    ax_corr_proc.set_title(f"'{variable_name}' 변화에 따른 공정별 불량률 (기본 vs 상관관계 반영)")
    ax_corr_proc.legend()
    ax_corr_proc.grid(True)

    # 반환
    return fig_proc, fig_total, fig_corr_proc





# ?? 페이지 구성 함수 정의
def page_home():
   st.title("?? 홈")
   st.markdown("""
   ### ?? 반도체 패키징 공정이란?
   반도체 패키징 공정은 웨이퍼에서 개별 칩을 분리하고 외부 환경으로부터 보호하면서 전기적 연결을 제공하는 일련의 공정입니다.
   각 공정은 제품의 신뢰성과 성능에 중대한 영향을 미칩니다.
   아래는 주요 공정의 흐름도입니다.
   """)

   # 이미지 보여주기
   safe_image("images/Package.JPG", caption="반도체 패키징 공정 전체 흐름", use_container_width=True)
   st.markdown("""
   ---
   ### ?? 주요 공정 설명

   - **Backlap (백래핑)**: 웨이퍼의 뒷면을 연마해 두께를 조절하고 스트레스를 해소하는 공정입니다.
   - **Sawing (쏘잉)**: 개별 칩을 잘라내는 공정입니다.
   - **Die Attach (다이 어태치)**: 잘라낸 칩을 패키지 기판에 부착합니다.
   - **Wire Bonding (와이어 본딩)**: 칩과 기판을 금속 와이어로 연결해 전기적 신호를 전달합니다.
   - **Molding (몰딩)**: 칩을 보호하기 위해 수지로 밀봉합니다.
   - **Marking (마킹)**: 제품 정보나 로고 등을 인쇄합니다.
   이러한 공정 중 어느 하나라도 최적 조건을 벗어나면 불량률이 증가하게 됩니다.
   본 시스템은 각 공정의 변수들을 입력 받아, 예측 모델을 통해 **불량률을 추정**하고 시각화해줍니다.

   ### ?? 주요 기능
   
   가상 실험 환경 제공: 실제 공정 데이터를 기반으로 구성된 변수 및 상관관계를 반영하여, 다양한 조건에서의 불량률을 시뮬레이션할 수 있습니다.

   공정 최적화 연습: 변수 조절을 통해 불량률을 최소화하는 조건을 탐색할 수 있으며, 이는 향후 공정 개선 방향 설정에 도움을 줍니다.

   공정 간 상관관계 학습: 각 공정의 결과가 다음 단계에 어떤 영향을 주는지 직관적으로 이해할 수 있도록 설계되어 공정 엔지니어에게 이해도를 높입니다.
   """)

def page_process_variable_info():
    st.title("?? 공정별 변수 설정 및 근거")

    # 1단계: 공정 선택 메뉴
    공정 = st.selectbox("공정을 선택하세요", [
        "Back Grinding (백래핑)",
        "Sawing (쏘잉)",
        "Die Attach (다이 어태치)",
        "Wire Bonding (와이어 본딩)",
        "Molding (몰딩)",
        "Marking (마킹)"
    ])

    if 공정 == "Back Grinding (백래핑)":
        st.header("?? 백래핑 공정 변수 및 근거")
        st.markdown("""
        - **웨이퍼 두께 (150?450 μm)**:
        - **연삭 속도 (25?100 rpm)**: 
        - **냉각수 유량 (5?20 L/min)**: 
        - **연삭 압력 (10?50 N)**: 
        """)
        safe_image("images/thickness_speed.JPG", caption="백래핑 공정 변수 설정 근거", use_container_width=True)
        safe_image("images/coolant_LAP_pressure.JPG", caption="백래핑 공정 변수 설정 근거", use_container_width=True)

    elif 공정 == "Sawing (쏘잉)":
        st.header("?? 쏘잉 공정 변수 및 근거")
        st.markdown("""
        - **블레이드 두께 (15,60)(um)**:
        - **웨이퍼 이송속도(7,15) (mm/s)**:
        - **블레이드 회전속도(30,110) (m/s)**:
        - **냉각수 유량(8,20) (L/min)**:          
        """)
        safe_image("images/sawing.JPG", caption="쏘잉 공정 변수 설정 근거", use_container_width=True)
        
    elif 공정 == "Die Attach (다이 어태치)":
         st.header("?? 다이 어태치 공정 변수 및 근거")
         st.markdown("""
         - **접착 온도(150,200)(℃)**:
         - **접착 압력(10,50) (Mpa)**:
         - **접착 시간(10,60) (s)**:
         - **접착제 점도(1000,4000)(Pa·s)**:            
         """)
         safe_image("images/Die_temp.JPG", caption="다이 어태치 공정 변수 설정 근거", use_container_width=True) 
         safe_image("images/Die.JPG", caption="다이 어태치 공정 변수 설정 근거", use_container_width=True)
    
    elif 공정 == "Wire Bonding (와이어 본딩)":
        st.header("?? 와이어 본딩 공정 변수 및 근거")
        st.markdown("""
        - **와이어 두께(203,381)um**:
        - **1차 본드 하중 (35,50)gm**:
        - **1차 본드 초음파 (35,50)Mw**:
        - **1차 본드 시간 (10,20)ms**:
        - **2차 본드 하중 (90~130)gm**:
        - **2차 본드 초음파 (90~130)Mw**:
        - **2차 본드 시간 (10~20)ms**:
        - **초음파 주파수 (100,150) kHz**:            
        """)
        safe_image("images/Wire.JPG", caption="와이어 본딩 공정 변수 설정 근거", use_container_width=True)
        safe_image("images/ultra_freq.JPG", caption="와이어 본딩 공정 변수 설정 근거", use_container_width=True)
    
    elif 공정 == "Molding (몰딩)":
         st.header("?? 몰딩 공정 변수 및 근거")
         st.markdown("""
         - **몰드 온도 : 150~200°C**:
         - **몰드 압력 : 80 ~ 140bar**:
         - **몰딩 시간 : 200 ~ 300s**:
         - **몰드 레진 점도 : 10? ~ 10?Pa·s**:            
         """)
         safe_image("images/Mold_temp.JPG", caption="몰딩 공정 변수 설정 근거", use_container_width=True)
         safe_image("images/Mold_time.JPG", caption="몰딩 공정 변수 설정 근거", use_container_width=True)
         safe_image("images/resin_viscosity.JPG", caption="몰딩 공정 변수 설정 근거", use_container_width=True)
    
    elif 공정 == "Marking (마킹)":
        st.header("?? 마킹 공정 변수 및 근거")
        st.markdown("""
        - **레이저 출력: 13 ~ 20W**:
        - **펄스 주파수: 10 ~ 50kHz**:
        - **마킹 속도: 67 ~ 200mm/s**:
        - **마킹 깊이: 16 ~ 72μm**:            
        """)
        safe_image("images/mark.JPG", caption="마킹 공정 변수 설정 근거", use_container_width=True)
        safe_image("images/mark_speed.JPG", caption="마킹 공정 변수 설정 근거", use_container_width=True)
        
def page_process_variable_correlation_info():
    st.title("?? 공정별 변수 상관관계 근거")

    # 1단계: 공정 선택 메뉴
    공정 = st.selectbox("공정을 선택하세요", [
        "Back Grinding (백래핑)",
        "Sawing (쏘잉)",
        "Die Attach (다이 어태치)",
        "Wire Bonding (와이어 본딩)",
        "Molding (몰딩)",
        "Marking (마킹)"
    ])

    if 공정 == "Back Grinding (백래핑)":
        st.header("?? 백래핑 공정 변수 상관관계 근거")
        st.markdown("""
        - **웨이퍼 두께 (150?450 μm)**:
        - **연삭 속도 (25?100 rpm)**: 
        - **냉각수 유량 (5?20 L/min)**: 
        - **연삭 압력 (10?50 N)**: 
        """)
        safe_image("images/backlap1.JPG", caption="백래핑 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/backlap2.JPG", caption="백래핑 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/backlap3.JPG", caption="백래핑 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/backlap4.JPG", caption="백래핑 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/backlap5.JPG", caption="백래핑 공정 변수 상관관계 근거", use_container_width=True)

    elif 공정 == "Sawing (쏘잉)":
        st.header("?? 쏘잉 공정 변수 상관관계 근거")
        st.markdown("""
        - **블레이드 두께 (15,60)(um)**:
        - **웨이퍼 이송속도(7,15) (mm/s)**:
        - **블레이드 회전속도(30,110) (m/s)**:
        - **냉각수 유량(8,20) (L/min)**:          
        """)
        safe_image("images/sawing1.JPG", caption="쏘잉 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/sawing2.JPG", caption="쏘잉 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/sawing3.JPG", caption="쏘잉 공정 변수 상관관계 근거", use_container_width=True)
        
    elif 공정 == "Die Attach (다이 어태치)":
         st.header("?? 다이 어태치 공정 변수 상관관계 근거")
         st.markdown("""
         - **접착 온도(150,200)(℃)**:
         - **접착 압력(10,50) (Mpa)**:
         - **접착 시간(10,60) (s)**:
         - **접착제 점도(1000,4000)(Pa·s)**:            
         """)
         safe_image("images/dieattach1.JPG", caption="다이 어태치 공정 변수 상관관계 근거", use_container_width=True) 
         safe_image("images/dieattach2.JPG", caption="다이 어태치 공정 변수 상관관계 근거", use_container_width=True)
    
    elif 공정 == "Wire Bonding (와이어 본딩)":
        st.header("?? 와이어 본딩 공정 변수 상관관계 근거")
        st.markdown("""
        - **와이어 두께(203,381)um**:
        - **1차 본드 하중 (35,50)gm**:
        - **1차 본드 초음파 (35,50)Mw**:
        - **1차 본드 시간 (10,20)ms**:
        - **2차 본드 하중 (90~130)gm**:
        - **2차 본드 초음파 (90~130)Mw**:
        - **2차 본드 시간 (10~20)ms**:
        - **초음파 주파수 (100,150) kHz**:            
        """)
        safe_image("images/wirebond1.JPG", caption="와이어 본딩 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/wirebond2.JPG", caption="와이어 본딩 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/wirebond3.JPG", caption="와이어 본딩 공정 변수 상관관계 근거", use_container_width=True)
    
    elif 공정 == "Molding (몰딩)":
         st.header("?? 몰딩 공정 변수 상관관계 근거")
         st.markdown("""
         - **몰드 온도 : 150~200°C**:
         - **몰드 압력 : 80 ~ 140bar**:
         - **몰딩 시간 : 200 ~ 300s**:
         - **몰드 레진 점도 : 10? ~ 10?Pa·s**:            
         """)
         safe_image("images/mold1.JPG", caption="몰딩 공정 변수 상관관계 근거", use_container_width=True)
         safe_image("images/mold2.JPG", caption="몰딩 공정 변수 상관관계 근거", use_container_width=True)
         safe_image("images/mold3.JPG", caption="몰딩 공정 변수 상관관계 근거", use_container_width=True)
         safe_image("images/mold4.JPG", caption="몰딩 공정 변수 상관관계 근거", use_container_width=True)
         safe_image("images/mold5.JPG", caption="몰딩 공정 변수 상관관계 근거", use_container_width=True)
    
    elif 공정 == "Marking (마킹)":
        st.header("?? 마킹 공정 변수 상관관계 근거")
        st.markdown("""
        - **레이저 출력: 13 ~ 20W**:
        - **펄스 주파수: 10 ~ 50kHz**:
        - **마킹 속도: 67 ~ 200mm/s**:
        - **마킹 깊이: 16 ~ 72μm**:            
        """)
        safe_image("images/mark1.JPG", caption="마킹 공정 변수 상관관계 근거", use_container_width=True)
        safe_image("images/mark2.JPG", caption="마킹 공정 변수 상관관계 근거", use_container_width=True)


def page_prediction():
    st.title("🔍 불량률 예측")
    st.markdown("총 40개 이상의 공정 변수를 입력하면, 일부 변수는 상관관계에 따라 자동으로 보정됩니다.")

    df = pd.read_csv("data/가상_공정_데이터.csv")
    models = load_models()

    # 1. 초기값 설정
    for col in input_cols:
        if col not in st.session_state:
            min_val, max_val = range_dict[col]
            st.session_state[col] = float((min_val + max_val) / 2)

    # 2. 입력 위젯 표시
    changed_vars = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(input_cols):
        min_val, max_val = range_dict[col]

        with (col1 if i % 2 == 0 else col2):
            new_val = st.number_input(
                f"{col} ({min_val}~{max_val})",
                min_value=float(min_val),
                max_value=float(max_val),
                key=col
            )
            if abs(new_val - st.session_state[col]) > 1e-6:
                changed_vars[col] = new_val

    # 3. 상관관계 보정
    adjusted_values = {col: st.session_state[col] for col in input_cols}
    for changed_col, changed_val in changed_vars.items():
        correlation_updates = apply_correlation(changed_col, changed_val, adjusted_values)
        adjusted_values.update(correlation_updates)

    # 4. 예측 버튼
    if st.button("📊 불량률 예측하기"):
        user_input = [adjusted_values[col] for col in input_cols]
        st.session_state["adjusted_input"] = adjusted_values.copy()
        st.session_state["last_user_input"] = user_input  # 조정 제안에 사용될 입력 저장

        result = predict_all(user_input, df, models)

        st.success(f"✅ 최종 공정 불량률: {result['final_defect']*100:.4f}%")
        for col in target_cols:
            st.write(f"📌 {col}: {result[col]*100:.4f}%")

        if len(changed_vars) > 0:
            st.markdown("---")
            st.subheader("🛠️ 자동 보정된 변수들:")
            for col in input_cols:
                original = st.session_state[col]
                adjusted = adjusted_values[col]
                if abs(original - adjusted) > 1e-6:
                    st.write(f"🔄 **{col}**: 입력값 {original:.4f} → 보정값 {adjusted:.4f}")

    # 5. 조정 제안 버튼
    st.markdown("---")
    if st.button("💡 조정 제안 보기"):
        if "last_user_input" in st.session_state:
            suggestions = suggest_adjustments(models, st.session_state["last_user_input"])
            st.subheader("📌 최적 변수 값 제안 (실제 영향 기준)")
            for col in target_cols:
                if col in suggestions:
                    s = suggestions[col]
                    st.markdown(f"""
                    **{col}**
                    - 영향 큰 변수: `{s['variable']}`
                    - 현재 값: `{s['current']:.2f}`
                    - 최적 값: `{s['optimal']:.2f}`
                    - 제안: {s['suggestion']}
                    """)
        else:
            st.warning("⚠️ 먼저 '불량률 예측하기' 버튼을 클릭해 주세요.")
def page_analysis():
    st.title("?? 특정 공정 분석")
    df = pd.read_csv("data/가상_공정_데이터.csv")
    models = load_models()

    st.subheader("?? 변수별 불량률 영향도")
    selected_var = st.selectbox("불량률 그래프를 보고 싶은 변수 선택", input_cols)

    # 사용자 입력 슬라이더 생성
    user_input = []
    for col in input_cols:
        min_val, max_val = range_dict[col]

        default_val = st.session_state.get("adjusted_input", {}).get(col, float((min_val + max_val) / 2))

        if col == selected_var:
            val = st.slider(
                f"{col} 값 선택",
                min_value=float(min_val),
                max_value=float(max_val),
                value=default_val,
                step=(max_val - min_val) / 100
            )
        else:
            val = default_val

        user_input.append(val)

    if st.button("?? 그래프 보기"):
        fig_proc, fig_total, fig_corr_proc = plot_defect_trend(selected_var, user_input, df, models)
        st.markdown("**?? 전체 불량률 기준 그래프**")
        st.pyplot(fig_total)
        st.markdown("**?? 해당 공정 불량률 기준 그래프**")
        st.pyplot(fig_proc)
        st.markdown("**?? 상관관계 반영 해당 공정 불량률 그래프**")
        st.pyplot(fig_corr_proc)

    st.markdown("---")
    st.subheader("?? 2D 변수 시각화")

    all_columns = input_cols + detail_cols + target_cols + ["final_defect"]
    x_var = st.selectbox("X축 선택", all_columns, index=0)
    y_var = st.selectbox("Y축 선택", all_columns, index=1)

    if st.button("?? 2D 분포 시각화"):
        # x축 값 생성
        if x_var in df.columns:
            x_vals = df[x_var].values
        else:
            x_vals = np.array([
                predict_all(row[input_cols].values.tolist(), df, models)[x_var]
                for _, row in df.iterrows()
            ])

        # y축 값 생성
        if y_var in df.columns:
            y_vals = df[y_var].values
        else:
            y_vals = np.array([
                predict_all(row[input_cols].values.tolist(), df, models)[y_var]
                for _, row in df.iterrows()
            ])

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(x_vals, y_vals, c=np.array(df["final_defect"]) * 100, cmap='plasma', alpha=0.6)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"{x_var} vs {y_var} (컬러: 최종 불량률 %)")
        plt.colorbar(sc, label="최종 불량률 (%)")

        # 입력값 기준 위치
        result = predict_all(user_input, df, models)
        user_x = user_input[input_cols.index(x_var)] if x_var in input_cols else result[x_var]
        user_y = user_input[input_cols.index(y_var)] if y_var in input_cols else result[y_var]
        ax.scatter([user_x], [user_y], color='red', s=100, edgecolors='black', label='입력값 위치')
        ax.legend()
        st.pyplot(fig)

        st.markdown("### ?? 입력값 기준 출력")
        if x_var not in input_cols:
            st.write(f"?? {x_var}: {user_x * 100:.4f}%" if "defect" in x_var else f"{x_var}: {user_x:.4f}")
        if y_var not in input_cols:
            st.write(f"?? {y_var}: {user_y * 100:.4f}%" if "defect" in y_var else f"{y_var}: {user_y:.4f}")

        st.markdown("### ?? 전체 불량률 요약")
        for col in target_cols:
            st.write(f"{col}: {result[col]*100:.4f}%")
        st.success(f"?? 총 불량률: {result['final_defect']*100:.4f}%")



###  메인 함수: 페이지 선택 구조 추가

def main():
    st.sidebar.title("메뉴")
    page = st.sidebar.selectbox("이동할 페이지 선택", ["홈", "공정별 변수 설정 및 근거","공정별 변수 상관관계 근거","불량률 예측", "특정 공정 분석"])

    if page == "홈":
        page_home()
    elif page == "공정별 변수 설정 및 근거":
        page_process_variable_info()
    elif page == "공정별 변수 상관관계 근거":
        page_process_variable_correlation_info()
    elif page == "불량률 예측":
        page_prediction()
    elif page == "특정 공정 분석":
        page_analysis()

if __name__ == "__main__":
    main()
