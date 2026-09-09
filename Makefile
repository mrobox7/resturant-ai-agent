.PHONY: dev ui test graph

dev:
	uv run uvicorn app.main:app --reload

ui:
	uv run streamlit run ui/streamlit_app.py

test:
	uv run pytest -v

graph:
	uv run python -c "from app.agents.graph import render_graph; print(render_graph())"
