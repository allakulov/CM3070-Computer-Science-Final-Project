# Procurement Document Enrichment System

**CM3070 Computer Science Final Project**

## Problem

Government e-procurement systems publish structured data about procurement procedures but the substantive details are locked inside attached documents in various formats. For my final project, I will build a multi-agent system that extracts structured data from procurement documents using multiple pre-trained AI models.

## Data

The system uses Latvian public procurement data from [eis.gov.lv](https://www.eis.gov.lv), with structured metadata available as open data on [data.gov.lv](https://data.gov.lv/dati/eng/dataset/izsludinato-iepirkumu-datu-grupa).

Each procurement procedure publishes multiple documents. Expected formats include PDF, possibly scanned PDFs (e.g., certificates), DOCX (templates), and XLSX (e.g. technical specifications, financial proposals). Documents are primarily expected to be in Latvian, so I will need to experiment with directly working on multi-lingual models or translating the documents first.

## Approach

At this early stage, I foresee an agent orchestrating specialised extraction agents, each using different pre-trained models across different data domains. The system will use at least three models, for example:

1. A local language model for classifying document types
2. An OCR model for scanned document pages
3. A document layout or table detection model for tabular data

The orchestrator classifies each document, routes it to the appropriate agent, and merges the partial extractions into one enriched record per procurement process.

## Plan

1. **Data acquisition**: download procurement documents from eis.gov.lv, either manually or by scraping. If I manage to automate document downloads, I can work with a larger corpus, otherwise I will need to keep the manual work managable.
2. **Agent development**: build extraction agents for each document format, backed by pre-trained models
3. **Orchestration**: connect agents into a multi-agent workflow.
4. **Evaluation**: compare model alternatives, measure extraction accuracy against manually annotated ground truth.

## Resources: 
1. Langchain [tools](https://docs.langchain.com/oss/python/langchain/tools)
2. Langgraph [tutorial](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph)
3. Middleware such as human-in-the-loop, model fallback, etc - see [here](https://docs.langchain.com/oss/python/langchain/middleware/built-in)