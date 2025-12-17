# Note
- 이 `AlphaApple` 프로젝트는 `사과게임`을 강화학습을 통해서 해결하는 모델을 만드는 프로젝트임
- `README.md` 는 브레인스토밍과 작업, 프로젝트 진행에 따라 업데이트 되는 "살아있는 문서"로 사용할 것

## Environment Contract (IMPORTANT)

- 본 프로젝트의 RL 환경은 `envs/fruitbox_env_improved.py`를 **단일 진실 소스(single source of truth)** 로 사용한다.
- LLM agent는 새로운 Env 클래스를 임의로 생성하거나, observation / action space를 변경해서는 안 된다.
- 환경 수정이 필요한 경우:
  1. 반드시 기존 env를 먼저 읽고
  2. 수정 이유를 README.md에 문서화한 뒤
  3. 최소 변경 원칙으로 반영한다.

## RL Strategy Guardrail

- 최초 목표는 "baseline 수립"이다.
- 복잡한 알고리즘(PPO, SAC 등)은 baseline(DQN or simple policy) 이후에만 도입한다.
- reward shaping, state augmentation은 baseline 실패 원인을 분석한 뒤 진행한다.
- 새로운 알고리즘 도입 시:
  - 왜 기존 방법이 부족한지 명시할 것

# Coding Convention
- commit은 중요하다고 생각되는 분기마다 자주 할 것
- commit message는 한국어로 작성할 것
- 파일 구조 세분화는 좋으나 과한 세분화는 피할 것
- 작업 내용에 대해 문서화할 것
- uv 환경을 사용하고 있음 `source .venv/bin/activate` 로 가상환경 활성화
- 실험은 기본적으로 .ipynb jupter notebook 파일로 진행할 것임
- 하지만 .ipynb jupter notebook 을 작성하고 읽는 것은 context 낭비가 심하기에
- .py python 코드를 #%% 문법을 사용하여 작성하고, jupytext로 .py->.ipynb 동기화 진행할 것
- **중요**: .py 파일 작성/수정 후에는 반드시 `jupytext --to notebook <file>.py` 명령어를 실행하여 .ipynb 파일을 생성/동기화해야 함 (단순 테스트용 스크립트 제외)
- **Git 규칙**: `.ipynb` 파일은 git에 커밋하지 않음 (바이너리 충돌 방지 및 용량 관리). 오직 `.py` 파일만 커밋함.

## #%% 사용법
```python
#%% [markdown]
# this is markdown
```

```python
#%% [code]
print("this is code")
```

## Code Ownership Rules

- 실험적 코드: `experiments/` 하위에서만 작성
- 재사용 가능한 로직 (model, trainer, utils): `src/` 하위에 작성
- 실험 결과를 근거 없이 src로 옮기는 행위 금지
- src 코드 변경 시 반드시 README.md에 변경 이유 요약

# EXPERIMENT ENVIRONMENT
- 사용자의 local 환경은 macbook pro m4 16gb ram 을 사용중 (mps 가속 사용 가능)
- 이 local 한경으로 실험하기 힘들 것 같은 환경이면 colab에서 notebook을 실행할 것임
- 그 경우 colab에 맞도록 코드를 작성해야함

# Reference
- [사과게임 링크](https://www.gamesaien.com/game/fruit_box_a/)
- [사과게임 나무위키](https://namu.wiki/w/フルーツボックス)
- `./envs/fruitbox_env_improved.py`에 구현된 환경을 참고할 것