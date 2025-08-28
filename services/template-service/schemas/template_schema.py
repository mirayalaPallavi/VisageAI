from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from datetime import datetime

# Base schemas
class TemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None
    is_active: bool = True

class TemplateVersionBase(BaseModel):
    version_number: str
    file_path: str
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    is_current: bool = False

class TemplateMetadataBase(BaseModel):
    key: str
    value: str
    data_type: str = "string"

class TemplateCategoryBase(BaseModel):
    name: str
    description: Optional[str] = None
    parent_id: Optional[int] = None
    is_active: bool = True

class TemplateTagBase(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = None

# Create schemas
class TemplateCreate(TemplateBase):
    pass

class TemplateVersionCreate(TemplateVersionBase):
    template_id: int

class TemplateMetadataCreate(TemplateMetadataBase):
    template_id: int

class TemplateCategoryCreate(TemplateCategoryBase):
    pass

class TemplateTagCreate(TemplateTagBase):
    pass

# Update schemas
class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None
    is_active: Optional[bool] = None

class TemplateVersionUpdate(BaseModel):
    version_number: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    is_current: Optional[bool] = None

class TemplateMetadataUpdate(BaseModel):
    key: Optional[str] = None
    value: Optional[str] = None
    data_type: Optional[str] = None

# Response schemas
class TemplateVersionResponse(TemplateVersionBase):
    id: int
    template_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class TemplateMetadataResponse(TemplateMetadataBase):
    id: int
    template_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class TemplateCategoryResponse(TemplateCategoryBase):
    id: int
    created_at: datetime
    children: List['TemplateCategoryResponse'] = []
    
    class Config:
        from_attributes = True

class TemplateTagResponse(TemplateTagBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class TemplateResponse(TemplateBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    template_versions: List[TemplateVersionResponse] = []
    template_metadata: List[TemplateMetadataResponse] = []
    
    class Config:
        from_attributes = True

# List response schemas
class TemplateListResponse(BaseModel):
    templates: List[TemplateResponse]
    total: int
    page: int
    size: int
    pages: int

# Search schemas
class TemplateSearchParams(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None
    page: int = 1
    size: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"

# Filter schemas
class TemplateFilter(BaseModel):
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    file_types: Optional[List[str]] = None
