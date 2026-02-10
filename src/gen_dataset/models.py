from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from enum import Enum

class StatementTypeEnum(str, Enum):
    """Whether a statement is normative, descriptive, or correlative"""
    NORMATIVE = "Normative" # Harmful/Stereotypical
    DESCRIPTIVE = "Descriptive" # Legally Valid/Factual
    CORRELATIVE = "Correlation" # Correlational/Relational

class Statement(BaseModel):
    """A statement combining EMGSD concepts with difference awareness context."""
    id: str = Field(description="Unique identifier for the statement instance", required=False)
    type: StatementTypeEnum = Field(description="The form of the statement: Normative or Descriptive or Correlative")
    marked_text: str = Field(description="Text with critical tokens marked", required=True)

class StatementMetadata(BaseModel):
    """Additional metadata for a statement."""
    statement: Statement = Field(description="Statement object holding marked text and type", required=True)
    cleaned_text: str = Field(description="The main text of the statement  (no markers)", required=True)
    themes: list[str] = Field(description="Themes or context relevant to the statement")
    target_groups: list[str] = Field(description="The stereotyped or protected groups that the statement refers to")
    data_source: str = Field(description="Where the statement has been adapted from")
    # sources: list[str] = Field(description="References or relevant external sources for the statement")
    is_harmful: bool = Field(description="Final classification of whether the statement is harmful or not", required=True)

class GeneratedStatements(BaseModel):
    """Response model for multiple statements."""
    statements: list[Statement]

class BADStatement(BaseModel):
    """A statement in the [acronym] dataset."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the statement instance", required=False)
    type: StatementTypeEnum = Field(description="The form of the statement: Normative, Descriptive, or Correlative")
    cleaned_text: Optional[str] = Field(None, description="The main text of the statement (no markers)", required=False)
    marked_text: str = Field(description="Text with critical tokens marked", required=True)
    themes: list[str] = Field(description="Themes or context relevant to the statement")
    target_groups: list[str] = Field(description="The stereotyped or protected groups that the statement refers to")
    data_source: str = Field(description="Where the statement has been adapted from")
    is_harmful: bool = Field(description="Final classification of whether the statement is harmful or not", required=True)