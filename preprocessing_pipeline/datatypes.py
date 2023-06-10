from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class Attribute:
    name: str
    value: str
    type: str

@dataclass
class Edge:
    id: str  # Added id field
    source: str
    target: str
    type: str

@dataclass
class Node:
    id: str
    content: str
    type: str
    attributes: Dict[str, Attribute]
    
@dataclass
class Document:
    id: str
    nodes: List[Node]
    edges: List[Edge]
    
@dataclass
class Root:
    documents: List[Document]
