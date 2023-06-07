from dataclasses import dataclass

@dataclass
class Attribute:
    name: str
    value: str
    type: str

@dataclass
class Edge:
    source: str
    target: str
    type: str
    attributes: Dict[str, Attribute]

@dataclass
class Node:
    id: str
    content: str
    type: str
    attributes: Dict[str, Attribute]
    edges: List[Edge]
    parent_id: Union[str, None] = None