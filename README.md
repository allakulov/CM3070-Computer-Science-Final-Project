# Procurement Document Enrichment System

**CM3070 Computer Science Final Project**

## Problem

Government e-procurement systems publish structured data about procurement procedures, but substantive details are locked inside attached documents in various formats. For my final project, I built a locally run system that extracts structured data from these documents using pre-trained AI models, deterministic checks, and human review.

The outputs can help check and enrich published records. They do not overwrite the official data.

## Data

The system uses Latvian public procurement data from [eis.gov.lv](https://www.eis.gov.lv), with structured metadata available as open data on [data.gov.lv](https://data.gov.lv/dati/eng/dataset/izsludinato-iepirkumu-datu-grupa).

Each procurement procedure publishes multiple documents. The reading layer supports digital and scanned PDFs, DOC/DOCX, XLS/XLSX, images, and text files, including documents inside nested ZIP, eDOC, and 7z containers. Legacy Word files are converted using LibreOffice.

Documents are primarily in Latvian. The models work directly with the source language, and extracted criterion names should retain the original wording.

## Approach

The system combines three kinds of locally run models:

1. A language model served through **Ollama** for interpreting procurement evidence.
2. **RapidOCR** for reading scanned content.
3. Table detection and structure recognition models for extracting tables from scanned pages.

**LangGraph** connects the reading, extraction, and validation steps through shared state. **Pydantic** schemas define the required structure of model responses.

The system extracts:

- **CPV codes:** main and additional codes, selected from candidates found in the documents and checked by a second model call.
- **Lots and award criteria:** lot information is extracted first. One criteria worker is assigned to each observed lot, searches for relevant prose and table rows, and extracts criterion names and weights. A fallback worker handles unknown lot scope.
- **Standards and related requirements:** patterns identify candidate standards, certificates, and legal instruments, retaining supporting excerpts for subsequent review.

Criteria weights are checked within each lot. Missing weights or inconsistent totals trigger one correction attempt with feedback. A sole criterion with no stated weight receives a recorded default only when its scoring group is identified.

Standards review runs separately from extraction. Clear cases can receive an automatic decision; unclear evidence, model failures, or missing decisions require human input. The command-line interface and Streamlit app use the same review logic.

## Project stages

1. **Data acquisition:** collect procurement metadata and download the associated attachments.
2. **Document reading:** extract text and tables while retaining source filenames and archive paths.
3. **Extraction and grounding:** coordinate model calls, validate outputs, retain evidence, and support human review.
4. **Evaluation:** measure agreement with usable official values and manually verify recovery where official fields are missing.

The final extraction evaluation covers 25 procurements. Macro scores give each procurement equal influence; complementary micro scores pool individual items. Criterion names are reviewed for equivalent meaning, not identical wording.

## Running the system

Run commands from the project directory with its Python environment activated, dependencies installed, and Ollama running.

Extract one procurement whose documents are in `downloads/162182/`:

```bash
python extract_graph.py --id 162182 --output-root runs/example
```

Run the saved 25-procurement sample:

```bash
python extract_graph.py --ids-file sample_25_ids.txt --output-root runs/sample_25
```

Review extracted standards:

```bash
python review_standards.py --id 162182 --extracted-dir runs/example/extracted
```

Review decisions are saved into the extraction files. Keep a backup before reviewing if you need the original outputs.

Launch the optional interface:

```bash
python -m streamlit run streamlit_app.py
```

Run the main test suite:

```bash
python -m unittest test_pipeline -v
```

## Outputs and limitations

Each procurement produces a structured JSON record, with separate table and OCR outputs. Logs and criteria traces support inspection of the evidence supplied to the models.

`validate_extraction.py` produces evaluation details and summaries. The export and Plotly modules provide JSON/CSV datasets and a visualization of official field coverage plus additions from the prototype.

Structured output and correct weight totals do not guarantee factual correctness. Remaining difficulties include incomplete evidence, conflicting documents, table alignment, and confusion between row numbers, weights, main criteria, and subcriteria. Outputs should be checked against their supporting evidence before use.

## Resources

1. [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
2. [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
3. [LangChain human-in-the-loop review](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
4. [Ollama documentation](https://docs.ollama.com/quickstart)