from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional
import structlog

from db import get_db
from models.db_models import Template, TemplateVersion, TemplateMetadata, TemplateCategory, TemplateTag
from schemas.template_schema import (
    TemplateCreate, TemplateUpdate, TemplateResponse, TemplateListResponse,
    TemplateVersionCreate, TemplateVersionUpdate, TemplateVersionResponse,
    TemplateMetadataCreate, TemplateMetadataUpdate, TemplateMetadataResponse,
    TemplateCategoryCreate, TemplateCategoryResponse,
    TemplateTagCreate, TemplateTagResponse,
    TemplateSearchParams
)

logger = structlog.get_logger()
router = APIRouter(prefix="/templates", tags=["templates"])

# Template CRUD operations
@router.post("/", response_model=TemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    template: TemplateCreate,
    db: Session = Depends(get_db)
):
    """Create a new template"""
    try:
        db_template = Template(**template.dict())
        db.add(db_template)
        db.commit()
        db.refresh(db_template)
        
        logger.info("Template created successfully", template_id=db_template.id, name=db_template.name)
        return db_template
    except Exception as e:
        db.rollback()
        logger.error("Failed to create template", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create template")

@router.get("/", response_model=TemplateListResponse)
async def get_templates(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    name: Optional[str] = None,
    category: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get list of templates with pagination and filtering"""
    query = db.query(Template)
    
    if name:
        query = query.filter(Template.name.ilike(f"%{name}%"))
    if category:
        query = query.filter(Template.category == category)
    if is_active is not None:
        query = query.filter(Template.is_active == is_active)
    
    total = query.count()
    templates = query.offset(skip).limit(limit).all()
    
    return TemplateListResponse(
        templates=templates,
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )

@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific template by ID"""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: int,
    template_update: TemplateUpdate,
    db: Session = Depends(get_db)
):
    """Update a template"""
    db_template = db.query(Template).filter(Template.id == template_id).first()
    if not db_template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    update_data = template_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_template, field, value)
    
    try:
        db.commit()
        db.refresh(db_template)
        logger.info("Template updated successfully", template_id=template_id)
        return db_template
    except Exception as e:
        db.rollback()
        logger.error("Failed to update template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update template")

@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: int,
    db: Session = Depends(get_db)
):
    """Delete a template"""
    db_template = db.query(Template).filter(Template.id == template_id).first()
    if not db_template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        db.delete(db_template)
        db.commit()
        logger.info("Template deleted successfully", template_id=template_id)
    except Exception as e:
        db.rollback()
        logger.error("Failed to delete template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete template")

# Template Version operations
@router.post("/{template_id}/versions", response_model=TemplateVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_template_version(
    template_id: int,
    version: TemplateVersionCreate,
    db: Session = Depends(get_db)
):
    """Create a new version for a template"""
    # Verify template exists
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        db_version = TemplateVersion(**version.dict())
        db.add(db_version)
        db.commit()
        db.refresh(db_version)
        
        logger.info("Template version created successfully", template_id=template_id, version_id=db_version.id)
        return db_version
    except Exception as e:
        db.rollback()
        logger.error("Failed to create template version", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create template version")

@router.get("/{template_id}/versions", response_model=List[TemplateVersionResponse])
async def get_template_versions(
    template_id: int,
    db: Session = Depends(get_db)
):
    """Get all versions of a template"""
    versions = db.query(TemplateVersion).filter(TemplateVersion.template_id == template_id).all()
    return versions

# Template Metadata operations
@router.post("/{template_id}/metadata", response_model=TemplateMetadataResponse, status_code=status.HTTP_201_CREATED)
async def create_template_metadata(
    template_id: int,
    metadata: TemplateMetadataCreate,
    db: Session = Depends(get_db)
):
    """Create metadata for a template"""
    # Verify template exists
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        db_metadata = TemplateMetadata(**metadata.dict())
        db.add(db_metadata)
        db.commit()
        db.refresh(db_metadata)
        
        logger.info("Template metadata created successfully", template_id=template_id, metadata_id=db_metadata.id)
        return db_metadata
    except Exception as e:
        db.rollback()
        logger.error("Failed to create template metadata", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create template metadata")

@router.get("/{template_id}/metadata", response_model=List[TemplateMetadataResponse])
async def get_template_metadata(
    template_id: int,
    db: Session = Depends(get_db)
):
    """Get metadata for a template"""
    metadata = db.query(TemplateMetadata).filter(TemplateMetadata.template_id == template_id).all()
    return metadata

# Category operations
@router.post("/categories", response_model=TemplateCategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    category: TemplateCategoryCreate,
    db: Session = Depends(get_db)
):
    """Create a new template category"""
    try:
        db_category = TemplateCategory(**category.dict())
        db.add(db_category)
        db.commit()
        db.refresh(db_category)
        
        logger.info("Category created successfully", category_id=db_category.id, name=db_category.name)
        return db_category
    except Exception as e:
        db.rollback()
        logger.error("Failed to create category", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create category")

@router.get("/categories", response_model=List[TemplateCategoryResponse])
async def get_categories(
    db: Session = Depends(get_db)
):
    """Get all template categories"""
    categories = db.query(TemplateCategory).filter(TemplateCategory.is_active == True).all()
    return categories

# Tag operations
@router.post("/tags", response_model=TemplateTagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag: TemplateTagCreate,
    db: Session = Depends(get_db)
):
    """Create a new template tag"""
    try:
        db_tag = TemplateTag(**tag.dict())
        db.add(db_tag)
        db.commit()
        db.refresh(db_tag)
        
        logger.info("Tag created successfully", tag_id=db_tag.id, name=db_tag.name)
        return db_tag
    except Exception as e:
        db.rollback()
        logger.error("Failed to create tag", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create tag")

@router.get("/tags", response_model=List[TemplateTagResponse])
async def get_tags(
    db: Session = Depends(get_db)
):
    """Get all template tags"""
    tags = db.query(TemplateTag).all()
    return tags
