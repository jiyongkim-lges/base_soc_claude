# NCM 셀 EKF SOC 추정: 개발 계획

## 목표와 합격 기준

- 2RC 등가회로모델(ECM) 기반 EKF로 주행 패턴 시험 데이터의 SOC를 추정합니다.
- 초기 SOC 오차를 **+5%p와 −5%p**로 두고 시작해서 오차가 줄어드는 쪽으로 수렴해야 합니다.
- **합격 기준: 수렴 이후 |SOC 오차|의 최댓값(MaxAE) ≤ 3%p.** 모든 시험 온도와 주행 패턴에서 만족해야 합니다.
- 전류적산(Coulomb counting)만으로 동작하는 EKF를 걸러내는 장치:
  같은 ±5%p 초기 오차로 순수 전류적산도 함께 돌립니다. 전류적산은 5%p 편향을 그대로 유지하므로 MaxAE 3%를 통과할 수 없습니다.
  EKF가 통과하려면 전압 측정으로 SOC를 실제로 보정해야 합니다.

### MaxAE를 재는 구간 (확정, docs/adr/0002)

초기 오차가 5%p이므로 t=0부터 MaxAE를 재면 항상 5%p 이상이 나와서 기준을 만족할 수 없습니다. 그래서 아래처럼 정의합니다.

- **수렴 시간 t_conv**: |오차|가 처음으로 3%p 아래로 내려온 시점. 상한(예: 600 s)을 함께 둡니다.
- **MaxAE**: t ≥ t_conv 구간의 max |SOC_est − SOC_true|.
- 함께 기록하는 지표: RMSE, MAE, t_conv, 단자전압 추정 RMSE.

## 1단계: 데이터셋 준비 (완료, 결과: reports/01_dataset.md)

1단계에서 확인·결정한 사항:
- HPPC 펄스는 9.9 s, 0.1 s 기록이므로 R_1s와 R_10s(9.9 s 시점)를 모두 계산할 수 있습니다.
- SOC 기준 용량은 25 °C C/20 방전용량 3.0326 Ah 하나로 통일합니다(docs/adr/0001).
- 저온에서 저항이 전류에 크게 의존해서, 2RC 파라미터를 (SOC, 온도, |전류|) 3차원 표로 만들었습니다.
- OCV는 평균 곡선과 방전 측 곡선 두 가지를 두고, 2단계에서 SOC 오차로 고릅니다.
- −10 °C, −20 °C에서는 진짜 SOC를 넣어도 모델 전압 오차가 큽니다(RMSE 70–250 mV). 저온 SOC 합격이 가장 어려운 조건이 될 것입니다.

### 선택한 데이터셋

**Samsung INR21700-30T 3Ah (NCM), McMaster University, Dr. Phillip Kollmeyer, CC BY 4.0**
Mendeley Data `9xyvy2njj3` v2: <https://data.mendeley.com/datasets/9xyvy2njj3/2>

| 요구사항 | 데이터셋 내용 | 확인 상태 |
| --- | --- | --- |
| 1.1 HPPC로 R0, R1, τ1, R2, τ2 추출 | 4-pulse 방전 HPPC(1, 2, 6, 12C), 충전 펄스(0.5, 1, 2, 4C), SOC 100, 95, 90, 80…20, 15, 10, 5, 2.5% | 페이지에서 확인함 |
| 1.2 1초, 10초 저항 | HPPC 펄스에서 계산 | **펄스 길이와 샘플링 주기는 다운로드 후 확인 필요** |
| 1.3 주행 패턴 + 실제 SOC | UDDS, HWFET, LA92, US06, Mix1–8 | 페이지에서 확인함 |
| 1.3 다양한 온도 | 40, 25, 10, 0, −10, −20 °C | 페이지에서 확인함 |
| OCV-SOC | C/20 충방전 시험 | 페이지에서 확인함 |

참고: 요청하신 온도 목록의 "25도, 25도"는 중복으로 보고, 0·10·25·−10 °C를 필수로, 40·−20 °C를 추가 검증용으로 씁니다.

### 작업

1. 다운로드한 원본은 `data/raw/`에 두고 git에서 제외합니다. MATLAB `.mat` 로더로 공통 스키마(`t, I, V, T_cell, Ah`)로 변환해 `data/processed/`에 저장합니다.
2. **OCV-SOC 곡선**: 온도별 C/20 충방전 곡선의 평균으로 만듭니다(히스테리시스는 절반으로 평균). HPPC의 휴지 종료 전압으로 교차 검증합니다.
3. **ECM 파라미터(1.1)**: 각 온도×SOC 펄스마다
   - R0 = 펄스 시작 순간의 ΔV / ΔI
   - R1, τ1, R2, τ2 = 펄스 후 휴지 구간의 이완 전압을 2-지수 함수로 `scipy.optimize.curve_fit` 피팅 (τ1 < τ2 로 정렬)
   - 결과를 (SOC, T) 2차원 룩업 테이블로 저장합니다.
4. **1초, 10초 저항(1.2)**: R_1s = (V(t0) − V(t0+1 s)) / ΔI, R_10s = (V(t0) − V(t0+10 s)) / ΔI. 온도×SOC 표와 그래프로 정리합니다.
5. **실제 SOC(1.3)**: 각 주행 시험 시작 직전의 만충 상태를 SOC 100%로 두고, 같은 온도에서 측정한 용량으로 전류를 적산합니다(시험기 정확도 0.1% FS). 종료 시점 전압과 OCV 곡선으로 적산 오차를 점검합니다.

### 산출물

- `reports/01_dataset.md`: 시험 목록, 온도·패턴 매트릭스, 비어 있는 항목
- `params/ecm_2rc.csv`, `params/ocv_soc.csv`, `params/r1s_r10s.csv`
- 2RC 모델 단독 전압 시뮬레이션 RMSE (EKF 이전에 모델 정확도를 먼저 확인)

## 2단계: EKF 구현과 평가

- 상태: x = [SOC, V1, V2]ᵀ, 입력: 전류 I (방전 +), 출력: 단자전압.
- 상태방정식(이산, Δt):
  - SOC[k+1] = SOC[k] − η·Δt·I[k] / (3600·Q_cap(T))
  - Vi[k+1] = exp(−Δt/τi)·Vi[k] + Ri·(1 − exp(−Δt/τi))·I[k],  i = 1, 2
- 출력방정식: V = OCV(SOC, T) − V1 − V2 − R0·I
- 야코비안 H = [dOCV/dSOC, −1, −1]. 파라미터는 (SOC, T) 룩업에서 매 스텝 보간합니다.
- 실행 매트릭스: {25, 10, 0, −10 °C (필수) + 40, −20 °C} × {UDDS, HWFET, LA92, US06, Mix} × {초기 오차 +5%p, −5%p}.
- 각 실행마다 SOC_true, SOC_EKF, SOC_CC(전류적산), 오차 그래프와 지표 표를 `reports/`에 저장합니다.
- 코드: `src/soc_ekf/` 아래에 `data`, `ocv`, `ecm`, `ekf`, `metrics`, `tuning` 모듈. `pytest`로 검증합니다(합성 데이터에서 알려진 파라미터 재현, 노이즈 없는 경우 EKF 수렴).

## 3단계: Q/R 조정 반복

1. 기준값(문헌 가이드라인)
   - R: 전압 센서 노이즈 분산과 2RC 모델 전압 잔차 분산 중 큰 값.
   - Q: SOC 항은 전류 센서 오차를 SOC 단위로 환산한 작은 값, V1·V2 항은 SOC 항보다 2~4 자릿수 큰 값에서 시작합니다(Mößle 외, 2023).
2. 2단계 실행 → 합격 기준 확인 → 실패하면 4단계 최적화로 Q/R을 다시 정하고 2단계를 반복합니다.
3. 반복해도 실패하면 Q/R보다 모델이 원인일 가능성이 큽니다. 다음 순서로 진단합니다: 단자전압 잔차 → 저온 OCV/R0 정확도 → 히스테리시스 → 용량의 온도 의존성.

## 4단계: Q/R 최적화 방법 (문헌 기반)

### 목적함수

J(Q, R) = max over 학습 세트 [ MaxAE_post_conv ] + w · mean(t_conv)

- +5%p와 −5%p 초기 오차 실행을 모두 포함하므로 전류적산형 해(Q_SOC → 0, 큰 R)는 자동으로 불리해집니다.
- 탐색 변수: log10(Q_SOC), log10(Q_V1), log10(Q_V2), log10(R). 로그 공간에서 탐색합니다.

### 학습/검증 분리 (과적합 방지)

- 학습: UDDS, US06 (전 온도)
- 검증: HWFET, LA92, Mix1–8
- 최종 보고는 검증 세트 결과로 합니다.

### 방법과 적용 순서

| 순서 | 방법 | 근거 | 역할 |
| --- | --- | --- | --- |
| 1 | 베이지안 최적화 (Optuna TPE 또는 GP) | Chen 외, "Weak in the NEES?: Auto-tuning Kalman Filters with Bayesian Optimization" (FUSION 2018) | 기본 방법. EKF 1회 실행 비용이 크므로 적은 평가 횟수로 좋은 Q/R을 찾음 |
| 2 | 메타휴리스틱 (GA, PSO, HHO) | HHO 기반 EKF 튜닝 논문(Energies 2024), GA/진화 알고리즘(Ting 외) | BO 결과 교차 검증용 비교 기준 |
| 3 | 다목적 최적화 (NSGA-II) | Mößle, Tietze, Danzer, Energy Technology (2023) | 정확도와 수렴 속도의 Pareto 곡선을 보고 운영점 선택 |
| 4 | 자가공분산 최소제곱(ALS) | ALS / MM-EKF SOC 논문 | 데이터의 innovation으로 R을 추정해 탐색 범위를 좁힘 |
| 5 | 적응형 EKF (Sage-Husa, 공분산 매칭) | 적응형 필터 문헌 | 고정 Q/R로 전 온도 합격이 안 될 때의 대안 |

고정 Q/R 하나로 모든 온도를 만족하지 못하면, 온도별 Q/R 스케줄(룩업)로 확장한 다음 적응형 EKF를 검토합니다.

## 진행 순서와 산출물

| 단계 | 산출물 | 완료 조건 |
| --- | --- | --- |
| 0 | Python 환경, 폴더 구조, pytest | `pytest` 통과 |
| 1 | 데이터 로더, OCV, ECM 파라미터, 1s/10s 저항 표 | 2RC 전압 RMSE 보고 |
| 2 | EKF와 평가 리포트 | 전체 실행 매트릭스 지표 표 |
| 3–4 | Q/R 최적화와 최종 리포트 | 검증 세트 전 조건 MaxAE ≤ 3%p |

## 참고 자료

- Samsung INR21700-30T 데이터: <https://data.mendeley.com/datasets/9xyvy2njj3/2>
- Mößle 외, Kalman Filter Tuning for State Estimation of Lithium-Ion Batteries by Multi-Objective Optimization (2023): <https://epub.uni-bayreuth.de/7352/>
- ALS / MM-EKF SOC 추정: <https://trid.trb.org/View/1412673>
- Q/R 공분산 적응 일반론: <https://arxiv.org/pdf/1702.00884>
- G. L. Plett, Extended Kalman filtering for battery management systems of LiPB-based HEV battery packs, Parts 1–3, J. Power Sources 134 (2004)
- Z. Chen 외, Weak in the NEES?: Auto-tuning Kalman Filters with Bayesian Optimization, FUSION 2018
