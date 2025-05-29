import numpy as np
import pandas as pd


# ▶️ 공통 함수
def linear_defect(val, max_val, min_val, d_min, d_max):
    return np.interp(val, [min_val, max_val], [d_max, d_min])

def gaussian_defect(val, mean, min_val, max_val, d_max):
    std_dev = (max_val - min_val) / 6
    prob = np.exp(-0.5 * ((val - mean) / std_dev) ** 2)
    return d_max * (1 - prob)

# ▶️ 백래핑 공정
def backlap_defect(thickness, speed, coolant, LAP_pressure, fab_temp, fab_humidity):
    bounds = {
        "thickness": (0.0006, 0.0045),
        "speed":     (0.00045, 0.003375),
        "LAP_pressure":  (0.00045, 0.003375),
        "coolant":   (0.0003, 0.00225),
        "fab_temp":      (0.00014, 0.00105),
        "fab_humidity":  (0.00006, 0.00045)
    }
    corrected_thickness = thickness - 0.03 * (LAP_pressure - 30) - 0.02 * (speed - 60)
    corrected_thickness += 0.36 * (fab_temp - 22)
    corrected_coolant = coolant + 0.15 * (speed - 60) + 0.2 * (fab_humidity - 40)

    d_thickness = linear_defect(corrected_thickness, 450, 150, *bounds["thickness"])
    d_speed = linear_defect(speed, 100, 25, *bounds["speed"])
    d_LAP_pressure = gaussian_defect(LAP_pressure, 30, 10, 50, bounds["LAP_pressure"][1])
    d_coolant  = gaussian_defect(corrected_coolant, 12, 5, 20, bounds["coolant"][1])
    d_fab_temp  = gaussian_defect(fab_temp, 22, 20, 24, bounds["fab_temp"][1])
    d_fab_humidity = gaussian_defect(fab_humidity, 40, 35, 45, bounds["fab_humidity"][1])

    contribs = {
        "backlap_thickness_defect": d_thickness,
        "backlap_speed_defect": d_speed,
        "backlap_LAP_pressure_defect": d_LAP_pressure,
        "backlap_coolant_defect": d_coolant,
        "backlap_temp_defect": d_fab_temp,
        "backlap_humidity_defect": d_fab_humidity,
    }
    total = sum(contribs.values())
    return np.clip(total, 0.002, 0.015), contribs

# 쏘잉 공정
def sawing_defect(blade_thickness, feed_rate, blade_speed, coolant_flow, SAW_fab_temp, SAW_fab_humidity):
    # ▸ 상관관계 반영: 습도 증가 → feed_rate 감소
    adjusted_feed_rate = feed_rate - 0.1 * (SAW_fab_humidity - 40)  # 기준 습도 40%
    adjusted_feed_rate = np.clip(adjusted_feed_rate, 7, 15)  # feed_rate 유효 범위 유지

    weights = {
        "blade_thickness": 0.15, "feed_rate": 0.25, "blade_speed": 0.25,
        "coolant_flow": 0.25, "SAW_fab_temp": 0.05, "SAW_fab_humidity": 0.05
    }
    SAWING_DEFECT_MIN = 0.002
    SAWING_DEFECT_MAX = 0.02
    min_contrib = {k: weights[k] * SAWING_DEFECT_MIN for k in weights}
    max_contrib = {k: weights[k] * SAWING_DEFECT_MAX for k in weights}

    def gaussian(x, mean, min_val, max_val, d_min, d_max):
        std = (max_val - min_val) / 6
        coeff = (x - mean) / std
        normalized = np.exp(-0.5 * coeff**2)
        return d_max - (d_max - d_min) * normalized

    def productivity_boost(feed_rate, base=11, max_reduction=0.3, max_feed=15):
        if feed_rate <= base:
            return 1.0
        reduction = max_reduction * (feed_rate - base) / (max_feed - base)
        return 1.0 - min(reduction, max_reduction)

    def blade_speed_defect(speed): return (30 - speed) * (78 - 75) / (110 - 30) + 78
    def feed_rate_defect(rate): return (rate - 7) * (80.5 - 75) / (15 - 7) + 75

    defect_contributions = {}
    defect_contributions["sawing_blade_thickness_defect"] = gaussian(blade_thickness, 37.5, 15, 60, min_contrib["blade_thickness"], max_contrib["blade_thickness"])
    
    feed_defect = (feed_rate_defect(adjusted_feed_rate) - feed_rate_defect(7)) / (feed_rate_defect(15) - feed_rate_defect(7))
    d_feed = min_contrib["feed_rate"] + feed_defect * (max_contrib["feed_rate"] - min_contrib["feed_rate"])
    d_feed *= productivity_boost(adjusted_feed_rate)
    defect_contributions["sawing_feed_rate_defect"] = d_feed

    blade_defect = (blade_speed_defect(blade_speed) - blade_speed_defect(110)) / (blade_speed_defect(30) - blade_speed_defect(110))
    defect_contributions["sawing_blade_speed_defect"] = min_contrib["blade_speed"] + blade_defect * (max_contrib["blade_speed"] - min_contrib["blade_speed"])

    required_coolant = 0.04 * blade_thickness + 0.3 * adjusted_feed_rate + 0.03 * blade_speed
    coolant_deficit = max(0, required_coolant - coolant_flow)
    d_cool = gaussian(coolant_flow, 14, 8, 20, min_contrib["coolant_flow"], max_contrib["coolant_flow"])
    d_cool += 0.0005 * coolant_deficit
    defect_contributions["sawing_coolant_flow_defect"] = d_cool

    SAW_humid_defect = (SAW_fab_humidity - 35) / (45 - 35)
    defect_contributions["sawing_fab_humidity_defect"] = max_contrib["SAW_fab_humidity"] - SAW_humid_defect * (max_contrib["SAW_fab_humidity"] - min_contrib["SAW_fab_humidity"])
    defect_contributions["sawing_fab_temp_defect"] = gaussian(SAW_fab_temp, 22, 20, 24, min_contrib["SAW_fab_temp"], max_contrib["SAW_fab_temp"])

    total_defect_rate = sum(defect_contributions.values())
    return np.clip(total_defect_rate, SAWING_DEFECT_MIN, SAWING_DEFECT_MAX), defect_contributions


# ▶️ 다이 어태치 공정 (확장 포함: pressure, time, fab_temp)
def gaussian_defect_dieattach(x, mean, min_val, max_val, d_min, d_max):
    std = (max_val - min_val) / 6
    z = (x - mean) / std
    return d_min + (d_max - d_min) * (1 - np.exp(-0.5 * z * z))

def apply_dieattach_correlation(vars):
    base = {"Die_temp": 175, "Die_pressure": 25, "Die_time": 30, "viscosity": 2500}
    corr = vars.copy()
    corr["viscosity"] += 50 * (vars["DIE_fab_humid"] - 40)
    corr["Die_time"] += (22 - vars["DIE_fab_temp"])
    corr["viscosity"] -= 20 * (vars["Die_temp"] - base["Die_temp"])
    corr["Die_pressure"]  -= 0.5 * (vars["Die_temp"] - base["Die_temp"])
    corr["Die_time"]      -= 0.5 * (vars["Die_temp"] - base["Die_temp"])
    corr["Die_pressure"] += 0.005 * (corr["viscosity"] - base["viscosity"])
    corr["Die_time"]     += 0.001 * (corr["viscosity"] - base["viscosity"])
    corr["Die_time"]     -= 0.05 * (corr["Die_pressure"] - base["Die_pressure"])
    return corr

def dieattach_defect(Die_temp, Die_pressure, Die_time, viscosity, DIE_fab_temp, DIE_fab_humid):
    vars_in = {
        "Die_temp": Die_temp, "Die_pressure": Die_pressure, "Die_time": Die_time,
        "viscosity": viscosity, "DIE_fab_temp": DIE_fab_temp, "DIE_fab_humid": DIE_fab_humid
    }
    corr = apply_dieattach_correlation(vars_in)
    weights = {
        "Die_temp": 0.275, "Die_pressure": 0.20, "Die_time": 0.15,
        "viscosity": 0.275, "DIE_fab_temp": 0.035, "DIE_fab_humid": 0.065
    }
    DIEATTACH_DEFECT_MIN = 0.002
    DIEATTACH_DEFECT_MAX = 0.015
    min_contrib = {k: DIEATTACH_DEFECT_MIN * weights[k] for k in weights}
    max_contrib = {k: DIEATTACH_DEFECT_MAX * weights[k] for k in weights}
    ranges_die = {
        "Die_temp": (150, 200), "Die_pressure": (10, 50),
        "Die_time": (10, 60), "viscosity": (1000, 4000),
        "DIE_fab_temp": (20, 24), "DIE_fab_humid": (35, 45)
    }

    d_Die_temp = gaussian_defect_dieattach(corr["Die_temp"], 175, *ranges_die["Die_temp"], min_contrib["Die_temp"], max_contrib["Die_temp"])
    d_Die_pressure = gaussian_defect_dieattach(corr["Die_pressure"], 25, *ranges_die["Die_pressure"], min_contrib["Die_pressure"], max_contrib["Die_pressure"])
    d_Die_time = gaussian_defect_dieattach(corr["Die_time"], 30, *ranges_die["Die_time"], min_contrib["Die_time"], max_contrib["Die_time"])
    d_viscosity = gaussian_defect_dieattach(corr["viscosity"], 2500, *ranges_die["viscosity"], min_contrib["viscosity"], max_contrib["viscosity"])
    d_DIE_fabtemp =  gaussian_defect_dieattach(DIE_fab_temp,22, *ranges_die["DIE_fab_temp"], min_contrib["DIE_fab_temp"], max_contrib["DIE_fab_temp"])
    d_DIE_fabhumid =  gaussian_defect_dieattach(DIE_fab_humid,40, *ranges_die["DIE_fab_humid"], min_contrib["DIE_fab_humid"], max_contrib["DIE_fab_humid"])

    contribs = {
        "dieattach_temp_defect": d_Die_temp,
        "dieattach_pressure_defect": d_Die_pressure,
        "dieattach_time_defect": d_Die_time,
        "dieattach_viscosity_defect": d_viscosity,
        "dieattach_fab_temp_defect": d_DIE_fabtemp,
        "dieattach_fab_humid_defect": d_DIE_fabhumid
    }
    total = sum(contribs.values())
    return np.clip(total, DIEATTACH_DEFECT_MIN, DIEATTACH_DEFECT_MAX), contribs

 # ▶️ 와이어본딩 공정 함수 및 변수
WIRE_DEFECT_MIN = 0.002
WIRE_DEFECT_MAX = 0.04
wire_weights = {
    "wire_diameter": 14, "bond_force_1": 11, "bond_ultra_1": 11,
    "bond_time_1": 11, "bond_force_2": 11, "bond_ultra_2": 11,
    "bond_time_2": 11, "ultra_freq": 10, "Wire_fab_temp": 5, "Wire_fab_humidity": 5
}
total_wire_weight = sum(wire_weights.values())
wire_ranges = {
    "wire_diameter": (203, 381), "bond_force_1": (35, 50),
    "bond_ultra_1": (35, 50), "bond_time_1": (10, 20),
    "bond_force_2": (90, 130), "bond_ultra_2": (90, 130),
    "bond_time_2": (10, 20), "ultra_freq": (100, 150),
    "Wire_fab_temp": (20, 24), "Wire_fab_humidity": (35, 45)
}

def calculate_required_params(wire_diameter):
    ratio = (wire_diameter - 203) / (381 - 203)
    return {
        "bond_force_1": 35 + 15 * ratio,
        "bond_time_1": 10 + 10 * ratio,
        "bond_ultra_1": 35 + 15 * ratio,
        "bond_force_2": 90 + 40 * ratio,
        "bond_time_2": 10 + 10 * ratio,
        "bond_ultra_2": 90 + 40 * ratio,
    }

def linear_scale(value, min_val, max_val, min_defect, max_defect):
    return min_defect + (max_defect - min_defect) * (value - min_val) / (max_val - min_val)

def symmetric_linear_defect(freq, center=119, left=100, right=136, final_right=150,
                            min_defect=0.0002, max_defect=0.004):
    if freq <= center:
        return linear_scale(freq, left, center, max_defect, min_defect)
    elif freq <= right:
        return linear_scale(freq, center, right, min_defect, max_defect)
    else:
        return linear_scale(freq, right, final_right, max_defect, min_defect)

def wire_bonding_defect(inputs):
    wire_diameter_adj = inputs["wire_diameter"] + 0.36 * (inputs["Wire_fab_temp"] - 22)
    freq_adj = inputs["ultra_freq"] - 0.02 * (inputs["bond_force_1"] + inputs["bond_force_2"] - 125)

    required = calculate_required_params(wire_diameter_adj)
    sensitivity = 1 + 0.02 * abs(freq_adj - 119)

    defect_contributions = {}
    min_contrib = {k: wire_weights[k]/total_wire_weight * WIRE_DEFECT_MIN for k in wire_weights}
    max_contrib = {k: wire_weights[k]/total_wire_weight * WIRE_DEFECT_MAX for k in wire_weights}

    defect_contributions["wirebond_wire_diameter_defect"] = linear_scale(
        wire_diameter_adj, 203, 381, max_contrib["wire_diameter"], min_contrib["wire_diameter"]
    )

    for var in ["bond_force_1", "bond_time_1", "bond_ultra_1", "bond_force_2", "bond_time_2", "bond_ultra_2"]:
        req = required[var]
        diff = max(0, req - inputs[var])
        penalty = diff / (wire_ranges[var][1] - wire_ranges[var][0]) * (max_contrib[var] - min_contrib[var])
        penalty *= sensitivity
        defect_contributions[f"wirebond_{var}_defect"] = min_contrib[var] + penalty

    defect_contributions["wirebond_ultra_freq_defect"] = symmetric_linear_defect(
        freq_adj, min_defect=min_contrib["ultra_freq"], max_defect=max_contrib["ultra_freq"]
    )
    defect_contributions["wirebond_fab_temp_defect"] = linear_scale(
        inputs["Wire_fab_temp"], 20, 24, min_contrib["Wire_fab_temp"], max_contrib["Wire_fab_temp"]
    )
    defect_contributions["wirebond_fab_humidity_defect"] = linear_scale(
        inputs["Wire_fab_humidity"], 35, 45, max_contrib["Wire_fab_humidity"], min_contrib["Wire_fab_humidity"]
    )

    total_defect = sum(defect_contributions.values())
    return np.clip(total_defect, WIRE_DEFECT_MIN, WIRE_DEFECT_MAX), defect_contributions

#몰딩 공정
# 🎯 불량률 범위
MOLD_DEFECT_MIN = 0.002  # 0.2%
MOLD_DEFECT_MAX = 0.01   # 1.0%

# ✅ 가중치 설정
weights = {
    "Mold_temp": 27.5,
    "Mold_pressure": 22.5,
    "Mold_time": 15,
    "resin_viscosity": 25,
    "Mold_fab_temp": 5,
    "Mold_fab_humidity": 5
}
total_weight = sum(weights.values())

# ✅ 변수 범위
ranges = {
    "Mold_temp": (150, 200),
    "Mold_pressure": (80, 140),
    "Mold_time": (200, 300),
    "resin_viscosity": (1e7, 1e8),
    "Mold_fab_temp": (20, 24),
    "Mold_fab_humidity": (35, 45)
}

# ✅ 최적값
optimal = {
    "Mold_temp": 175,
    "Mold_pressure": 128,
    "Mold_time": 250,
    "resin_viscosity": 5e7,
    "Mold_fab_temp": 22,
    "Mold_fab_humidity": 40
}

# ✅ 가우시안 불량률 함수
def skew_gaussian_defect(value, mean, std, min_defect, max_defect, skew=False):
    if not skew:
        z = (value - mean) / std
    else:
        if value < mean:
            z = (value - mean) / (std * 1.8)  # 왼쪽 꼬리 더 넓게
        else:
            z = (value - mean) / (std * 0.8)  # 오른쪽 꼬리 좁게
    prob = np.exp(-0.5 * z ** 2)
    norm_prob = 1 - prob
    return min_defect + norm_prob * (max_defect - min_defect)

# ✅ 점도 기반 요구 압력/시간 산출
def calculate_required_conditions(viscosity):
    ratio = (viscosity - 1e7) / (1e8 - 1e7)
    req_Mold_pressure = 80 + 48 * ratio
    req_Mold_time = 200 + 100 * ratio
    return req_Mold_pressure, req_Mold_time

# ✅ 압력 기반 온도 민감도 보정
def temperature_sensitivity(Mold_pressure):
    return 1 + 0.03 * (128 - Mold_pressure)

# ✅ 전체 불량률 계산
def molding_defect(inputs):
    # 상관관계 1: FAB 습도 증가 → 점도 증가
    resin_viscosity_adj = inputs["resin_viscosity"] + 5e5 * (inputs["Mold_fab_humidity"] - 40)

    # 상관관계 2: 몰드 온도 증가 → 점도 감소 (1도 ↑당 2e5 감소)
    resin_viscosity_adj -= 2e5 * (inputs["Mold_temp"] - 175)

    # 범위 제한
    resin_viscosity_adj = np.clip(resin_viscosity_adj, *ranges["resin_viscosity"])

    # 상관관계 3: FAB 온도 감소 → 시간 증가 (1도 ↓당 10초 ↑)
    Mold_time_adj = inputs["Mold_time"] + 10 * (22 - inputs["Mold_fab_temp"])

    # 상관관계 4: 몰드 온도 증가 → 시간 감소 (1도 ↑당 2초 ↓)
    Mold_time_adj -= 2 * (inputs["Mold_temp"] - 175)

    # 범위 제한
    Mold_time_adj = np.clip(Mold_time_adj, *ranges["Mold_time"])

    req_Mold_pressure, req_Mold_time = calculate_required_conditions(resin_viscosity_adj)
    Mold_temp_sensitivity = temperature_sensitivity(inputs["Mold_pressure"])

    defect_contributions = {}
    min_contrib = {k: weights[k]/total_weight * MOLD_DEFECT_MIN for k in weights}
    max_contrib = {k: weights[k]/total_weight * MOLD_DEFECT_MAX for k in weights}

    defect_contributions["molding_Mold_temp_defect"] = gaussian_defect(
        inputs["Mold_temp"], optimal["Mold_temp"], 6 * Mold_temp_sensitivity,
        min_contrib["Mold_temp"], max_contrib["Mold_temp"]
    )

    defect_contributions["molding_Mold_pressure_defect"] = skew_gaussian_defect(
        inputs["Mold_pressure"], optimal["Mold_pressure"], 8,
        min_contrib["Mold_pressure"], max_contrib["Mold_pressure"], skew=True
    )

    defect_contributions["molding_Mold_time_defect"] = gaussian_defect(
        Mold_time_adj, optimal["Mold_time"], 20,
        min_contrib["Mold_time"], max_contrib["Mold_time"]
    )

    defect_contributions["molding_resin_viscosity_defect"] = gaussian_defect(
        resin_viscosity_adj, optimal["resin_viscosity"], 1.5e7,
        min_contrib["resin_viscosity"], max_contrib["resin_viscosity"]
    )

    # 벌점 계산
    Mold_pressure_penalty = max(0, req_Mold_pressure - inputs["Mold_pressure"])
    Mold_pressure_penalty *= (max_contrib["Mold_pressure"] - min_contrib["Mold_pressure"]) / (ranges["Mold_pressure"][1] - ranges["Mold_pressure"][0])
    defect_contributions["molding_Mold_pressure_defect"] += Mold_pressure_penalty

    Mold_time_penalty = max(0, req_Mold_time - Mold_time_adj)
    Mold_time_penalty *= (max_contrib["Mold_time"] - min_contrib["Mold_time"]) / (ranges["Mold_time"][1] - ranges["Mold_time"][0])
    defect_contributions["molding_Mold_time_defect"] += Mold_time_penalty

    defect_contributions["molding_Mold_fab_temp_defect"] = linear_scale(
        inputs["Mold_fab_temp"], 20, 24, max_contrib["Mold_fab_temp"], min_contrib["Mold_fab_temp"]
    )

    defect_contributions["molding_Mold_fab_humidity_defect"] = linear_scale(
        inputs["Mold_fab_humidity"], 35, 45, min_contrib["Mold_fab_humidity"], max_contrib["Mold_fab_humidity"]
    )

    total_defect = sum(defect_contributions.values())
    return np.clip(total_defect, MOLD_DEFECT_MIN, MOLD_DEFECT_MAX), defect_contributions

#마킹 공정
def marking_defect(
    mark_laser_power, mark_pulse_freq, mark_speed, mark_depth, mark_fab_temp, mark_fab_humidity
):
  
    # ▶️ 파라미터 설정 
    optimal_values = {
        'mark_laser_power': 16.5,
        'mark_pulse_freq': 30,
        'mark_speed': 130,
        'mark_depth': 33,
        'mark_fab_temp': 20,
        'mark_fab_humidity': 40
    }
    std_values = {
        'mark_laser_power': 1.4,
        'mark_pulse_freq': 8.9,
        'mark_speed': 29.5,
        'mark_depth': 12.4,
        'mark_fab_humidity': 2.2
    }

    weights = {
        'mark_laser_power': 27,
        'mark_pulse_freq': 24,
        'mark_depth': 21,
        'mark_speed': 21,
        'mark_fab_temp': 5,
        'mark_fab_humidity': 2
    }
    total_min_defect = 0.0001
    total_max_defect = 0.005
    skew_value = -1.5  # 마킹 깊이 전용

    # ▶️ 변수별 불량률 범위 계산
    defect_ranges = {
        k: (
            total_min_defect * (w / sum(weights.values())),
            total_max_defect * (w / sum(weights.values()))
        )
        for k, w in weights.items()
    }

    # ▶️ 불량률 계산 함수 정의
    def symmetric_gaussian(x, mean, std):
        z = (x - mean) / std
        return np.exp(-0.5 * z**2)

    def asymmetric_gaussian(x, mean, std, skew):
        z = (x - mean) / std
        norm = np.exp(-0.5 * z**2)  # 표준 가우시안
        asym = 1 / (1 + np.exp(-skew * z))  # 시그모이드 곡선
        return min(1.0, norm * asym * 2)

    def calculate_defect(variable, value, mean, std, skew=None):
        if skew is None:
            norm = symmetric_gaussian(value, mean, std)
        else:
            norm = asymmetric_gaussian(value, mean, std, skew)
        scaled = 1 - norm
        d_min, d_max = defect_ranges[variable]
        return d_min + (d_max - d_min) * scaled

    # ▶️ 변수 간 상관관계 반영
    mark_pulse_freq += 0.2 * (mark_laser_power - optimal_values['mark_laser_power'])
    mark_speed += 1.5 * (mark_laser_power - optimal_values['mark_laser_power'])
    mark_depth += 0.8 * (mark_laser_power - optimal_values['mark_laser_power'])

    mark_depth -= 0.4 * (mark_pulse_freq - optimal_values['mark_pulse_freq'])
    mark_speed += 0.3 * (mark_pulse_freq - optimal_values['mark_pulse_freq'])

    mark_depth -= 0.3 * (mark_speed - optimal_values['mark_speed'])

    delta_depth = 0.01 * (mark_fab_temp - 20)
    depth_variation = np.random.normal(0, delta_depth)
    mark_depth += depth_variation

    # ▶️ 각 변수별 불량률 계산
    defect_contrib = {}
    defect_contrib['marking_laser_power_defect'] = calculate_defect(
        'mark_laser_power', mark_laser_power, optimal_values['mark_laser_power'], std_values['mark_laser_power']
    )
    defect_contrib['marking_pulse_freq_defect'] = calculate_defect(
        'mark_pulse_freq', mark_pulse_freq, optimal_values['mark_pulse_freq'], std_values['mark_pulse_freq']
    )
    defect_contrib['marking_speed_defect'] = calculate_defect(
        'mark_speed', mark_speed, optimal_values['mark_speed'], std_values['mark_speed']
    )
    defect_contrib['marking_depth_defect'] = calculate_defect(
        'mark_depth', mark_depth, optimal_values['mark_depth'], std_values['mark_depth'], skew_value
    )
    # FAB 온도는 선형
    mark_fab_temp_norm = (mark_fab_temp - 20) / (24 - 20)
    d_min, d_max = defect_ranges['mark_fab_temp']
    defect_contrib['marking_fab_temp_defect'] = d_min + (d_max - d_min) * mark_fab_temp_norm

    # 습도는 가우시안
    defect_contrib['marking_fab_humidity_defect'] = calculate_defect(
        'mark_fab_humidity', mark_fab_humidity, optimal_values['mark_fab_humidity'], std_values['mark_fab_humidity']
    )

    # ▶️ 전체 불량률 계산
    total_defect = sum(defect_contrib.values())
    return np.clip(total_defect, total_min_defect, total_max_defect), defect_contrib

# ▶️ 가상 데이터 생성
n_samples = 10000
records = []

for _ in range(n_samples):
    v_back = {"thickness": np.random.uniform(150, 450),
              "speed": np.random.uniform(25, 100),
              "coolant": np.random.uniform(5, 20),
              "LAP_pressure": np.random.uniform(10, 50),
              "fab_temp": np.random.uniform(20, 24),
              "fab_humidity": np.random.uniform(35, 45)}
    v_saw = {"blade_thickness": np.random.uniform(15, 60),
             "feed_rate": np.random.uniform(7, 15),
             "blade_speed": np.random.uniform(30, 110),
             "coolant_flow": np.random.uniform(8, 20),
             "SAW_fab_temp": np.random.uniform(20, 24),
             "SAW_fab_humidity": np.random.uniform(35, 45)}
    v_die = {"Die_temp": np.random.uniform(150, 200),
             "Die_pressure": np.random.uniform(10, 50),
             "Die_time": np.random.uniform(10, 60),
             "viscosity": np.random.uniform(1000, 4000),
             "DIE_fab_temp": np.random.uniform(20, 24),
             "DIE_fab_humid": np.random.uniform(35, 45)}
    v_wire = {"wire_diameter": np.random.uniform(203, 381),
              "bond_force_1": np.random.uniform(35, 50),
              "bond_ultra_1": np.random.uniform(35, 50),
              "bond_time_1": np.random.uniform(10, 20),
              "bond_force_2": np.random.uniform(90, 130),
              "bond_ultra_2": np.random.uniform(90, 130),
              "bond_time_2": np.random.uniform(10, 20),
              "ultra_freq": np.random.uniform(100, 150),
              "Wire_fab_temp": np.random.uniform(20, 24),
              "Wire_fab_humidity": np.random.uniform(35, 45)}
    v_mold = {"Mold_temp": np.random.uniform(150, 200),
             "Mold_pressure": np.random.uniform(80, 140),
             "Mold_time": np.random.uniform(200, 300),
             "resin_viscosity": np.random.uniform(1e7, 1e8),
             "Mold_fab_temp": np.random.uniform(20, 24),
             "Mold_fab_humidity": np.random.uniform(35, 45)}
    v_mark = {"mark_laser_power": np.random.uniform(13, 20),
              "mark_pulse_freq": np.random.uniform(10, 50),
              "mark_speed": np.random.uniform(67, 200),
              "mark_depth": np.random.uniform(16, 72),
              "mark_fab_temp": np.random.uniform(20, 24),
              "mark_fab_humidity": np.random.uniform(35, 45)}

    d_back, c_back = backlap_defect(**v_back)
    d_saw, c_saw = sawing_defect(**v_saw)
    d_die, c_die = dieattach_defect(**v_die)
    d_wire, c_wire = wire_bonding_defect(v_wire)
    d_mold, c_mold = molding_defect(v_mold)
    d_mark, c_mark = marking_defect(**v_mark)
    
    row = {
        **v_back, **v_saw, **v_die, **v_wire, **v_mold, **v_mark,
        **c_back, **c_saw, **c_die, **c_wire, **c_mold, **c_mark,        
        "backlap_defect": d_back,
        "sawing_defect": d_saw,
        "dieattach_defect": d_die,
        "wirebond_defect": d_wire,
        "molding_defect": d_mold,
        "marking_defect": d_mark,
        "final_defect": 1 - (1 - d_back) * (1 - d_saw) * (1 - d_die) * (1 - d_wire) * (1 - d_mold) * (1 - d_mark)
    }
    records.append(row)

# ▶️ CSV 파일로 저장 + 소수점 10자리 유지
pd.DataFrame(records).to_csv("가상_공정_데이터.csv", index=False, float_format="%.10f")
