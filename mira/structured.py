"""Structured LLM output the way n8n's LangChain agents do it.

n8n's agents with a Structured Output Parser don't ask for raw JSON text: they
give the model a `format_final_json_response` tool whose parameters are the
parser's schema wrapped under `output`, and append a fixed instruction to the
system message. The provider then returns the answer as tool-call arguments,
which is far more robust than parsing free text (a long selection answer as
raw JSON text broke the weekly run). Schemas below are the live workflow's
parsers (execution-verified wording of the instruction).
"""
from __future__ import annotations

import json

TOOL_NAME = "format_final_json_response"
TOOL_DESCRIPTION = "Use this tool to format your final response to the user in a structured JSON format."
SYSTEM_SUFFIX = (
    "IMPORTANT: For your response to user, you MUST use the `format_final_json_response` tool with your "
    "complete answer formatted according to the required schema. Do not attempt to format the JSON "
    "manually - always use this tool. Your response will be rejected if it is not properly formatted "
    "through this tool. Only use this tool once you are ready to provide your final answer."
)

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

SCHEMAS = {
    # Structured Output Parser1 -> Parse Affiliations from PDF
    "affiliation": {
        "type": "object",
        "properties": {
            "arxiv_id": {"type": "string", "description": "The arXiv ID/URL of the paper for matching"},
            "affiliations": {**_STR_LIST, "description": "List of all unique institutions"},
            "author_affiliations": {"type": "object", "additionalProperties": _STR_LIST,
                                    "description": "Map of author name to their institutions"},
            "credibility_reasoning": _STR,
            "credibility_tier": {"type": "number"},
        },
        "required": ["arxiv_id", "affiliations", "author_affiliations", "credibility_reasoning",
                     "credibility_tier"],
    },
    # Structured Output Parser2 -> Topic Classification1
    "classification": {
        "type": "object",
        "properties": {
            "arxiv_id": _STR, "primary_topic": _STR, "secondary_topics": _STR_LIST,
            "potential_impact": _STR, "relevance_score": {"type": "number"}, "key_findings": _STR,
            "actionable": _STR,
        },
        "required": ["arxiv_id", "primary_topic", "secondary_topics", "potential_impact",
                     "relevance_score", "key_findings", "actionable"],
    },
    # Structured Output Parser (Selection): generated from its JSON example
    "selection": {
        "type": "object",
        "properties": {
            "reasoning": _STR,
            "selected_papers": {"type": "array", "items": {"type": "object", "properties": {
                "arxiv_id": _STR, "selection_reasoning": _STR, "priority_rank": {"type": "number"}}}},
            "remaining_papers": {"type": "array", "items": {"type": "object", "properties": {
                "arxiv_id": _STR, "exclusion_reasoning": _STR}}},
        },
    },
    # Structured Output Parser (Analysis)
    "analysis": {
        "type": "object",
        "properties": {"arxiv_id": _STR, "large_summary": _STR, "short_summary": _STR,
                       "pdf_analysis_performed": {"type": "boolean"}},
        "required": ["arxiv_id", "large_summary", "short_summary", "pdf_analysis_performed"],
    },
    # Structured Output Parser (Trend)
    "trend": {
        "type": "object",
        "properties": {"trend_section_markdown": _STR},
        "required": ["trend_section_markdown"],
    },
    # Media - Structured Output Parser (Selection): from its JSON example
    "media_selection": {
        "type": "object",
        "properties": {"selected_indices": {"type": "array", "items": {"type": "number"}}},
    },
}


def system_with_instruction(system: str) -> str:
    """n8n joins the agent's system message and the tool instruction with a
    blank line; an empty system message becomes just the instruction."""
    return f"{system}\n\n{SYSTEM_SUFFIX}" if system else SYSTEM_SUFFIX


def tool_for(schema: dict) -> dict:
    return {"type": "function", "function": {
        "name": TOOL_NAME, "description": TOOL_DESCRIPTION,
        "parameters": {"type": "object", "properties": {"output": schema}, "required": ["output"]}}}


def unwrap(arguments: str) -> str:
    """Tool-call arguments → the JSON text of the `output` object."""
    data = json.loads(arguments)
    if isinstance(data, dict) and "output" in data:
        data = data["output"]
        if isinstance(data, str):  # some models double-encode
            try:
                data = json.loads(data)
            except ValueError:
                return data
    return json.dumps(data, ensure_ascii=False)
