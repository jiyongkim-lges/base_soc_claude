# base_soc_claude

NCM 셀 SOC 추정의 기본 알고리즘인 EKF(Extended Kalman Filter)를 개발하는 저장소입니다.
전체 개발 계획은 `docs/plan/ekf-soc-dev-plan.md`에 있습니다.

## 실행 방법

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python scripts/download_30t.py   # 원본 데이터 (~57 MB, data/raw/, git 제외)
.venv/bin/python scripts/run_phase1.py     # params/, data/processed/, reports/01_dataset.md
.venv/bin/python -m pytest -q
```

## 규칙

- 전류 부호: 패키지 안에서는 **방전이 양수**입니다. 원본 파일은 방전이 음수입니다(`soc_ekf.data.load`에서 뒤집음).
- 전하량 적분은 우측값(right-hold)으로 합니다. 이 시험기는 각 샘플에 직전 구간의 전류를 기록합니다.
- SOC 정의와 합격 기준은 `docs/adr/`를 따릅니다.

## Agent skills

### Issue tracker

이슈는 GitHub Issues(`jiyongkim-lges/base_soc_claude`)에서 `gh` CLI로 관리합니다. See `docs/agents/issue-tracker.md`.

### Triage labels

기본 라벨 다섯 개(`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`)를 그대로 씁니다. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context 구조: 루트의 `GLOSSARY.md` 하나와 `docs/adr/`를 씁니다. See `docs/agents/domain.md`.
