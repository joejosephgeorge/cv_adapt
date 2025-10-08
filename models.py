"""
Pydantic models for CV Adaptor following the architecture document
These models serve as structured data contracts between agents
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Phase I: Input Data Models (Structured Extraction)
# ============================================================================

class ContactInfo(BaseModel):
    """Contact information from CV"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class Experience(BaseModel):
    """Work experience entry"""
    company: str
    position: str
    duration: str
    description: str
    achievements: List[str] = Field(default_factory=list)
    skills_used: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)  # Quantifiable metrics


class Education(BaseModel):
    """Education entry"""
    institution: str
    degree: str
    field: Optional[str] = None
    duration: str
    gpa: Optional[str] = None


class CandidateProfileSchema(BaseModel):
    """Structured data contract for CV content (Parser Agent output)"""
    contact: ContactInfo
    summary: Optional[str] = None
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class JobRequirement(BaseModel):
    """Individual job requirement"""
    category: str  # "Technical Skills", "Experience", "Education", etc.
    requirement: str
    importance: str  # "Required", "Preferred", "Nice to have"
    keywords: List[str] = Field(default_factory=list)


class JobRequirementSchema(BaseModel):
    """Structured data contract for job description (Parser Agent output)"""
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    requirements: List[JobRequirement] = Field(default_factory=list)
    key_skills: List[str] = Field(default_factory=list)
    experience_level: str  # "Entry", "Mid", "Senior", "Executive"
    industry: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    salary_range: Optional[str] = None
    benefits: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None  # For RAG indexing


# ============================================================================
# Phase II: RAG and Scoring Models
# ============================================================================

class SkillGap(BaseModel):
    """Individual skill gap identified"""
    skill: str
    importance: str
    present_in_cv: bool
    suggested_evidence: Optional[str] = None


class MatchGapReportSchema(BaseModel):
    """Output schema for Scoring Agent"""
    relevance_score: float = Field(ge=0.0, le=100.0)
    skill_gaps: List[SkillGap] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    target_keywords: List[str] = Field(default_factory=list)
    focus_areas: List[str] = Field(default_factory=list)
    recommendation: str  # "proceed", "optimize", "major_gaps"
    reasoning: Optional[str] = None  # Explainable AI component


# ============================================================================
# Phase III: Analysis Models
# ============================================================================

class SectionAnalysis(BaseModel):
    """Analysis for a specific CV section"""
    section_name: str  # "Summary", "Experience", "Skills", etc.
    current_status: str  # Brief assessment of current state
    items_to_add: List[str] = Field(default_factory=list)  # Specific items/points to add
    items_to_remove: List[str] = Field(default_factory=list)  # Specific items/points to remove
    items_to_modify: List[str] = Field(default_factory=list)  # Existing items to rephrase/modify
    keywords_to_add: List[str] = Field(default_factory=list)  # Keywords from JD to integrate
    priority: str = "medium"  # "high", "medium", "low"


class CVAnalysisReportSchema(BaseModel):
    """Concise CV Analysis Report organized by sections"""
    overall_assessment: str  # Brief overall assessment
    relevance_score: float = Field(ge=0.0, le=100.0)
    section_analyses: List[SectionAnalysis] = Field(default_factory=list)
    critical_gaps: List[str] = Field(default_factory=list)  # Must-have items missing
    strengths_to_emphasize: List[str] = Field(default_factory=list)  # What's already good
    quick_wins: List[str] = Field(default_factory=list)  # Easy improvements


# ============================================================================
# Phase IV: Quality Assurance Models
# ============================================================================

class QAIssue(BaseModel):
    """Individual QA issue identified"""
    section: str  # "summary", "experience", "skills", etc.
    issue_type: str  # "missing_keyword", "factual_error", "style_inconsistency"
    description: str
    severity: str  # "critical", "major", "minor"
    suggested_fix: Optional[str] = None


class QAReportSchema(BaseModel):
    """Output schema for QA Agent"""
    passed: bool
    overall_score: float = Field(ge=0.0, le=100.0)
    issues: List[QAIssue] = Field(default_factory=list)
    keywords_verified: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    factual_consistency_check: bool
    style_consistency_check: bool
    feedback_for_rewrite: Optional[str] = None


# ============================================================================
# Final Output Models
# ============================================================================

class CVAnalysisOutputSchema(BaseModel):
    """Final CV analysis output with all reports"""
    analysis_report: CVAnalysisReportSchema
    match_report: MatchGapReportSchema
    candidate_name: Optional[str] = None
    job_title: Optional[str] = None


# ============================================================================
# Workflow State Management (LangGraph)
# ============================================================================

class WorkflowState(BaseModel):
    """State for LangGraph workflow for CV analysis"""
    # Inputs
    cv_text: Optional[str] = None
    job_description: Optional[str] = None
    
    # Parsed data (Phase I)
    candidate_profile: Optional[CandidateProfileSchema] = None
    job_requirements: Optional[JobRequirementSchema] = None
    
    # RAG and scoring (Phase II)
    match_report: Optional[MatchGapReportSchema] = None
    
    # Analysis (Phase III)
    analysis_report: Optional[CVAnalysisReportSchema] = None
    
    # Final output
    final_output: Optional[CVAnalysisOutputSchema] = None
    
    # Workflow control
    current_step: str = "start"
    errors: List[str] = Field(default_factory=list)
    should_continue: bool = True
    
    # RAG context (for grounding)
    cv_rag_context: List[str] = Field(default_factory=list)
    jd_rag_context: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
        # Allow dict assignment for LangGraph compatibility
        validate_assignment = True