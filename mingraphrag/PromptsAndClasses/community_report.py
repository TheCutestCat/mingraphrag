from pydantic import BaseModel,Field
from typing import Literal
ENTITY_EXTRACTION_ENTITY_TYPES_list = ["PERSON", "GEO"]
ENTITY_EXTRACTION_ENTITY_TYPES_Literal = Literal["PERSON", "GEO"]

class Findings(BaseModel):
    summary : str
    explanation : str
    
    def to_str(self):
        return(f"summary : {self.summary}, explanation : {self.explanation}")

class CommunityReportResponseFormat(BaseModel):
    title: str
    summary : str
    rating : str
    rating_explanation : str
    findings : list[Findings]


CommunityReportPrompt = f"""
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

"""