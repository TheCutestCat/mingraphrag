from pydantic import BaseModel,Field
from typing import Literal
ENTITY_EXTRACTION_ENTITY_TYPES_list = ["PERSON", "GEO"]
ENTITY_EXTRACTION_ENTITY_TYPES_Literal = Literal["PERSON", "GEO"]

class Entity(BaseModel):
    entity_type: ENTITY_EXTRACTION_ENTITY_TYPES_Literal
    entity_name : str
    entity_description : str
    
    def tostr(self) -> str:
        return f"Entity(type={self.entity_type}, name={self.entity_name}, description={self.entity_description})"

    
class Relation(BaseModel):
    source_entity : str 
    target_entity: str 
    relationship_description : str
    relationship_strength : int
    
    def tostr(self) -> str:
        return (f"Relation(source={self.source_entity}, target={self.target_entity}, "
                f"description={self.relationship_description}, strength={self.relationship_strength})")


class EntityExtractionResponseFormat(BaseModel):
    entities : list[Entity]
    relations : list[Relation]
    
    def merge(self, other: 'EntityExtractionResponseFormat'):
        self.entities.extend(other.entities)  # 合并entities列表
        self.relations.extend(other.relations)  # 合并relations列表
        return self
    
    def get_all_entity_names(self):
        return [entity.entity_name for entity in self.entities]
    
    def tostr(self):
        entities_str = "\n".join(
            [f"Entity: {entity.entity_name}, Type: {entity.entity_type}, Description: {entity.entity_description}" 
             for entity in self.entities]
        )
        relations_str = "\n".join(
            [f"Relation: {relation.source_entity} -> {relation.target_entity}, "
             f"Description: {relation.relationship_description}, Strength: {relation.relationship_strength}" 
             for relation in self.relations]
        )
        return f"Entities:\n{entities_str}\n\nRelations:\n{relations_str}"

EntityExtractionPrompt = f"""
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
We will give a reference entity name, please use the entity names in the corresponding list in the relationship part.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized,
- entity_type: One of the following types: [{ENTITY_EXTRACTION_ENTITY_TYPES_list}]
- entity_description: Comprehensive description of the entity's attributes and activities
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity from 0 to 10
"""