from dataclasses import dataclass
from PIL import Image
from typing import List, Optional, Union
from pydantic import Field


@dataclass
class COCODatapointSegmentPolygons:
    polygons: List[List[float]]


@dataclass
class COCODatapointSegmentRLE:
    counts: List[int]
    size: List[int]


@dataclass
class COCODatapointSegmentRaw:
    segment: Union[COCODatapointSegmentPolygons, COCODatapointSegmentRLE]
    objectType: str


@dataclass
class COCODatapointRaw:
    image_path: str
    caption: str
    segments: List[COCODatapointSegmentRaw]
    bg_desc: Optional[str] = Field(default=None)


@dataclass
class COCODatapointSegment:
    mask: Image
    objectType: str


@dataclass
class COCODatapoint:
    image: Image
    caption: str
    segments: List[COCODatapointSegment]
    bg_desc: Optional[str] = Field(default=None)
