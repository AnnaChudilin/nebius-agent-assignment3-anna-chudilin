# Schemas for MCP Server API Endpoints
# Defines structured input models for tool execution payloads to ensure consistent parsing and validation.

from pydantic import BaseModel, Field


class FilterIntentInput(BaseModel):
    intent: str = Field(...,
                        description="Intent label to query, for example 'get_refund'.")


class ShowExamplesInput(BaseModel):
    category: str = Field(
        ..., description="Category label to show examples from, for example 'SHIPPING'.")
    n: int = Field(
        3, ge=1, le=10, description="Number of example records to return.")


class DistributionInput(BaseModel):
    category: str = Field(...,
                          description="Category for which to compute intent distribution.")


class SummarizeCategoryInput(BaseModel):
    category: str = Field(...,
                          description="Category for which to generate a high-level summary.")
