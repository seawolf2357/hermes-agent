---
title: Hermes Agent - Data Analysis
emoji: "\U0001F52E"
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# Hermes Agent - Data Analysis Demo

An interactive demo showcasing the data analysis capabilities of [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research.

## Features

- **AI Chat**: Converse with the Hermes Agent via an OpenAI-compatible API
- **Data Analysis**: Upload CSV/JSON files and ask natural language questions about your data
- **Visualization**: Auto-generated charts and statistical summaries
- **Code Display**: See the Python code the agent generates for analysis

## Configuration

Set the following environment variables in your Space settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for the LLM provider | (required) |
| `OPENAI_BASE_URL` | Base URL for the API endpoint | `https://openrouter.ai/api/v1` |
| `OPENAI_MODEL` | Model to use | `anthropic/claude-sonnet-4` |
