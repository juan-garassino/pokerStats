# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* pokerStats/*.py

black:
	@black scripts/* pokerStats/*.py


test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr deepSculpt-*.dist-info
	@rm -fr deepSculpt.egg-info

install:
	@pip install . -U

all: clean install test black check_code

# ----------------------------------
#          RL TRAINING
# ----------------------------------

train:
	python -m pokerStats.rl.self_play_trainer

train-render:
	python -m pokerStats.rl.self_play_trainer --render

eval-rl:
	python scripts/eval_rl.py

eval-rl-legacy:
	python scripts/eval_rl.py --legacy

# Colab training (configurable hand counts + table size)
PHASE1_HANDS ?= 20000
PHASE2_HANDS ?= 50000
PHASE3_HANDS ?= 20000
NUM_PLAYERS ?= 6
RENDER_EVERY ?= 5000

train-colab:
ifeq ($(USE_CFR),1)
	python -c "from pokerStats.rl.self_play_trainer import MultiAgentTrainer, CFG; \
		from pokerStats.cfr.distillation import BlueprintTeacher; \
		CFG['num_players']=$(NUM_PLAYERS); \
		CFG['phase1_hands']=$(PHASE1_HANDS); \
		CFG['phase2_hands']=$(PHASE2_HANDS); \
		CFG['phase3_hands']=$(PHASE3_HANDS); \
		CFG['render_every']=$(RENDER_EVERY); \
		teacher = BlueprintTeacher('$(CFR_BLUEPRINT)', '$(CFR_ABSTRACTION)'); \
		MultiAgentTrainer(CFG, blueprint_teacher=teacher).train()"
else
	python -c "from pokerStats.rl.self_play_trainer import MultiAgentTrainer, CFG; \
		CFG['num_players']=$(NUM_PLAYERS); \
		CFG['phase1_hands']=$(PHASE1_HANDS); \
		CFG['phase2_hands']=$(PHASE2_HANDS); \
		CFG['phase3_hands']=$(PHASE3_HANDS); \
		CFG['render_every']=$(RENDER_EVERY); \
		MultiAgentTrainer(CFG).train()"
endif

train-colab-quick:
	$(MAKE) train-colab PHASE1_HANDS=5000 PHASE2_HANDS=10000 PHASE3_HANDS=5000

train-colab-full:
	$(MAKE) train-colab PHASE1_HANDS=50000 PHASE2_HANDS=200000 PHASE3_HANDS=50000

# Table size presets — same network, different player counts
train-colab-2max:
	$(MAKE) train-colab NUM_PLAYERS=2

train-colab-3max:
	$(MAKE) train-colab NUM_PLAYERS=3

train-colab-6max:
	$(MAKE) train-colab NUM_PLAYERS=6

train-colab-8max:
	$(MAKE) train-colab NUM_PLAYERS=8

# ── Full hybrid agent training (PPO + CFR) ──────────────
# Best agent for a given table size — runs both pipelines
train-hybrid:
	$(MAKE) cfr-colab-full
	$(MAKE) train-colab-full NUM_PLAYERS=$(NUM_PLAYERS)

train-hybrid-quick:
	$(MAKE) cfr-colab-quick
	$(MAKE) train-colab-quick NUM_PLAYERS=$(NUM_PLAYERS)

train-hybrid-6max:
	$(MAKE) train-hybrid NUM_PLAYERS=6

train-hybrid-8max:
	$(MAKE) train-hybrid NUM_PLAYERS=8

# ── Unified training (CFR distillation into single network) ──
# Trains CFR blueprint first, then PPO with distillation loss
USE_CFR ?= 0
CFR_BLUEPRINT ?= checkpoints/cfr/blueprint.npz
CFR_ABSTRACTION ?= checkpoints/cfr/abstraction.npz

train-unified:
	$(MAKE) cfr-colab-full
	$(MAKE) train-colab-full NUM_PLAYERS=$(NUM_PLAYERS) USE_CFR=1

train-unified-quick:
	$(MAKE) cfr-colab-quick
	$(MAKE) train-colab-quick NUM_PLAYERS=$(NUM_PLAYERS) USE_CFR=1

train-unified-6max:
	$(MAKE) train-unified NUM_PLAYERS=6

train-unified-8max:
	$(MAKE) train-unified NUM_PLAYERS=8

# ── Step 1: CFR on CPU (cheap Colab instance, no GPU needed) ──
# Run this first, then `make zip-output` to download blueprint
# Benchmark: 5k iters = 4 hours on Colab CPU (0.5 iter/s)
cfr-cpu-quick:
	$(MAKE) cfr-train CFR_BUCKETS=20 CFR_SAMPLES=500 CFR_ITERS=500 CFR_RENDER=250 CFR_LOG=250

cfr-cpu:
	$(MAKE) cfr-train CFR_BUCKETS=20 CFR_SAMPLES=500 CFR_ITERS=1000 CFR_RENDER=500 CFR_LOG=500

cfr-cpu-full:
	$(MAKE) cfr-train CFR_BUCKETS=30 CFR_SAMPLES=500 CFR_ITERS=3000 CFR_RENDER=1000 CFR_LOG=1000

# ── Step 2: PPO+distillation on GPU (needs blueprint from step 1) ──
# Upload blueprint via zip-output from step 1, then run this
ppo-gpu-quick:
	$(MAKE) train-colab NUM_PLAYERS=$(NUM_PLAYERS) USE_CFR=1 \
		PHASE1_HANDS=5000 PHASE2_HANDS=10000 PHASE3_HANDS=5000

ppo-gpu:
	$(MAKE) train-colab NUM_PLAYERS=$(NUM_PLAYERS) USE_CFR=1 \
		PHASE1_HANDS=20000 PHASE2_HANDS=50000 PHASE3_HANDS=20000

ppo-gpu-full:
	$(MAKE) train-colab NUM_PLAYERS=$(NUM_PLAYERS) USE_CFR=1 \
		PHASE1_HANDS=50000 PHASE2_HANDS=200000 PHASE3_HANDS=50000

# ── All-in-one (if you want both on same instance) ──────
train-unified-colab-quick:
	$(MAKE) cfr-cpu-quick
	$(MAKE) ppo-gpu-quick

train-unified-colab:
	$(MAKE) cfr-cpu
	$(MAKE) ppo-gpu

train-unified-colab-full:
	$(MAKE) cfr-cpu-full
	$(MAKE) ppo-gpu-full

# Watch random agent play 10 hands (no checkpoint needed)
demo:
	python scripts/demo_play.py --hands 10 --delay 0.5 --players 3

# Watch with MCTS search enabled
demo-search:
	python scripts/demo_play.py --hands 10 --delay 0.5 --players 3 --search --simulations 50

# Watch random agent vs different opponents
demo-vs-maniac:
	python scripts/demo_play.py --hands 10 --opponent maniac --players 3

demo-vs-nit:
	python scripts/demo_play.py --hands 10 --opponent nit --players 3

demo-vs-tag:
	python scripts/demo_play.py --hands 10 --opponent tight-aggressive --players 4

# Watch a trained agent play (after training or downloading a checkpoint)
demo-trained:
	python scripts/demo_play.py --hands 10 --checkpoint checkpoints/best.pt --opponent fish --players 3

# Watch trained agent with MCTS
demo-trained-search:
	python scripts/demo_play.py --hands 10 --checkpoint checkpoints/best.pt --opponent fish --players 3 --search

# Full table 6-max demo
demo-6max:
	python scripts/demo_play.py --hands 5 --players 6 --delay 0.3

# ----------------------------------
#          CFR TRAINING
# ----------------------------------

CFR_DIR ?= checkpoints/cfr
CFR_BUCKETS ?= 50
CFR_SAMPLES ?= 1000
CFR_ITERS ?= 5000
CFR_RENDER ?= 500
CFR_LOG ?= 500
CFR_MIN_VISITS ?= 1

# Full CFR pipeline: abstraction → solve → blueprint
cfr-train: cfr-abstraction cfr-solve cfr-blueprint

cfr-abstraction:
	@mkdir -p $(CFR_DIR)
	python scripts/train_cfr.py --phase abstraction \
		--output-dir $(CFR_DIR) \
		--n-buckets $(CFR_BUCKETS) \
		--n-samples $(CFR_SAMPLES)

cfr-solve:
	python scripts/train_cfr.py --phase solve \
		--output-dir $(CFR_DIR) \
		--iterations $(CFR_ITERS) \
		--render-every $(CFR_RENDER) \
		--log-every $(CFR_LOG)

cfr-blueprint:
	python scripts/train_cfr.py --phase blueprint \
		--output-dir $(CFR_DIR) \
		--min-visits $(CFR_MIN_VISITS)

# Resume CFR training from checkpoint
cfr-resume:
	python scripts/train_cfr.py --phase solve \
		--output-dir $(CFR_DIR) \
		--iterations $(CFR_ITERS) \
		--render-every $(CFR_RENDER) \
		--log-every $(CFR_LOG) \
		--resume $(CFR_DIR)/solver_final.npz

# Presets (all include rendered demo hands)
cfr-colab-quick:
	$(MAKE) cfr-train CFR_BUCKETS=20 CFR_SAMPLES=500 CFR_ITERS=5000 CFR_RENDER=1000 CFR_LOG=500

cfr-colab:
	$(MAKE) cfr-train CFR_BUCKETS=50 CFR_SAMPLES=1000 CFR_ITERS=50000 CFR_RENDER=10000 CFR_LOG=5000

cfr-colab-full:
	$(MAKE) cfr-train CFR_BUCKETS=200 CFR_SAMPLES=10000 CFR_ITERS=500000 CFR_RENDER=50000 CFR_LOG=25000

# Test Kuhn poker convergence (instant sanity check)
cfr-test-kuhn:
	python -c "from pokerStats.cfr.cfr_solver import KuhnCFRSolver; \
		s = KuhnCFRSolver(); s.train(10000); \
		print('Kuhn Nash strategies:'); \
		[print(f'  {k:10s}  pass={v.average_strategy()[0]:.3f}  bet={v.average_strategy()[1]:.3f}') \
		 for k,v in sorted(s.info_sets.items())]"

# Run CFR tests only
cfr-test:
	python -m pytest tests/test_abstraction.py tests/test_cfr_solver.py tests/test_hybrid_agent.py -v --tb=short

# ── Zip repo (upload to Colab) ───────────────────────────
zip-repo:
	@rm -f pokerStats-repo.zip
	zip -r pokerStats-repo.zip \
		pokerStats/ scripts/ tests/ reference/ \
		Makefile setup.py requirements.txt CLAUDE.md \
		-x "*/__pycache__/*" "*.pyc" "*/.DS_Store"
	@echo ""
	@echo "  ✓ pokerStats-repo.zip"
	@ls -lh pokerStats-repo.zip

# ── Zip outputs (download from Colab) ───────────────────
zip-output:
	@rm -f pokerStats-output.zip
	zip -r pokerStats-output.zip \
		checkpoints/ data/ \
		-x "*/__pycache__/*"
	@echo ""
	@echo "  ✓ pokerStats-output.zip"
	@ls -lh pokerStats-output.zip

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------

PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      TRAIN MODEL
# ----------------------------------

# project id - replace with your GCP project id
PROJECT_ID=pokerStats

# bucket
BUCKET_NAME=pokerStats

# training folder
BUCKET_TRAINING_FOLDER=data

# training params, choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

# app environment
PYTHON_VERSION=3.7

FRAMEWORK=scikit-learn

RUNTIME_VERSION=2.2

# package params
PACKAGE_NAME=pokerStats

FILENAME=trainer

JOB_DIR=gs://pokerStats

MACHINE=config.yaml

MACHINE_GPU=config-gpu.yaml

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

run_locally:
	python -m pokerStats.trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=pokerStats_$(shell date +'%Y%m%d_%H%M%S')

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--config ${MACHINE} \
		--job-dir ${JOB_DIR} \
		--stream-logs

gcp_submit_training_gpu:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--config ${MACHINE_GPU} \
		--stream-logs

# ----------------------------------
#      RUN API
# ----------------------------------

api_run:
	python app.py
