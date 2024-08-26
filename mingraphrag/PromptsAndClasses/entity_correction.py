from pydantic import BaseModel,Field
from typing import Literal
ENTITY_EXTRACTION_ENTITY_TYPES_list = ["PERSON", "GEO"]
ENTITY_EXTRACTION_ENTITY_TYPES_Literal = Literal["PERSON", "GEO"]


EntityCorrectionPrompt = f"""
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, along with a list of descriptions and potential relationships between these entities,
please concatenate all of the descriptions into a single, comprehensive summary. Ensure that the summary includes 
information collected from all the descriptions and captures the relationships between entities accurately.

If the provided descriptions are contradictory, please resolve the contradictions to provide a single, coherent summary.
Ensure the summary is written in third person and includes the entity names for full context.

For each identified entity, extract and unify the following information:
- entity_name: Name of the entity, capitalized,
- entity_type: One of the following types: [{ENTITY_EXTRACTION_ENTITY_TYPES_list}]
- entity_description: Comprehensive description of the entity's attributes and activities

For relationships between entities, extract and unify the following information:
- source_entity: Name of the entity from which the relationship originates
- target_entity: Name of the entity to which the relationship is directed
- relationship_description: Detailed description of the relationship between the entities
- relationship_strength: Integer value representing the strength of the relationship
"""