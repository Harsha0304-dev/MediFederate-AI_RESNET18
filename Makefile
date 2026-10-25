install:
	python -m pip install -r requirements.txt
test:
	pytest -q
train:
	python scripts/train_xray.py
evaluate:
	python scripts/evaluate_xray.py
run:
	streamlit run app/streamlit_app.py
