.PHONY: all run_rag launch_ui train
all: run_rag launch_ui train

run_rag:
	bash ./src/CIT/RAGs/launch_RAG.sh

launch_ui_dev:
	streamlit run ./src/CIT/UI/streamlit_app_prod2.py

launch_ui_battle:
	streamlit run ./src/CIT/UI/streamlit_app_battle.py

launch_ui_prod:
	cd ./src/CIT/UI && \
	streamlit run streamlit_app_prod.py

scrape:
	python ./src/CIT/scraping/main.py