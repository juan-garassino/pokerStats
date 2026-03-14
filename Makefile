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

# Colab training (configurable hand counts)
PHASE1_HANDS ?= 20000
PHASE2_HANDS ?= 50000
PHASE3_HANDS ?= 20000

train-colab:
	python -c "from pokerStats.rl.self_play_trainer import MultiAgentTrainer, CFG; \
		CFG['phase1_hands']=$(PHASE1_HANDS); \
		CFG['phase2_hands']=$(PHASE2_HANDS); \
		CFG['phase3_hands']=$(PHASE3_HANDS); \
		CFG['render_every']=999999; \
		MultiAgentTrainer(CFG).train()"

train-colab-quick:
	$(MAKE) train-colab PHASE1_HANDS=5000 PHASE2_HANDS=10000 PHASE3_HANDS=5000

train-colab-full:
	$(MAKE) train-colab PHASE1_HANDS=50000 PHASE2_HANDS=200000 PHASE3_HANDS=50000

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
	$(MAKE) cfr-train CFR_BUCKETS=20 CFR_SAMPLES=500 CFR_ITERS=1000 CFR_RENDER=500 CFR_LOG=500

cfr-colab:
	$(MAKE) cfr-train CFR_BUCKETS=50 CFR_SAMPLES=1000 CFR_ITERS=5000 CFR_RENDER=500 CFR_LOG=500

cfr-colab-full:
	$(MAKE) cfr-train CFR_BUCKETS=200 CFR_SAMPLES=10000 CFR_ITERS=100000 CFR_RENDER=500 CFR_LOG=500

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
