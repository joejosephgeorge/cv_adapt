"""
Specialized LLM Agents for CV Analysis
Implements 3 key agents:
1. Parser Agent: Structured data extraction
2. Scoring Agent: RAG-enhanced relevance scoring
3. Analysis Agent: Generate concise CV analysis report by section
"""
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json
import re

from models import (
    CandidateProfileSchema,
    JobRequirementSchema,
    MatchGapReportSchema,
    CVAnalysisReportSchema,
    SectionAnalysis,
    SkillGap,
)
from rag_system import RAGSystem
from llm_factory import get_structured_llm


class ParserAgent:
    """
    Parser Agent: Extracts structured data from unstructured CV/JD text
    Uses high-reliability LLM with Pydantic schema enforcement and retry mechanism
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.max_retries = 3
    
    def parse_cv(self, cv_text: str) -> CandidateProfileSchema:
        """
        Parse CV text into structured CandidateProfileSchema
        Implements retry mechanism for failed Pydantic validation
        """
        prompt = ChatPromptTemplate.from_template("""
You are an expert CV parser. Extract structured information from the CV text below.
        
        CV Text:
        {cv_text}
        
Extract ALL information and return as valid JSON matching this schema:
        {{
            "contact": {{
                "name": "Full Name",
                "email": "email@example.com",
                "phone": "phone number",
                "location": "city, country",
        "linkedin": "linkedin URL",
        "github": "github URL"
            }},
    "summary": "Professional summary or objective",
            "experience": [
                {{
                    "company": "Company Name",
                    "position": "Job Title",
            "duration": "Start - End",
            "description": "Role description",
                    "achievements": ["achievement 1", "achievement 2"],
            "skills_used": ["skill1", "skill2"],
            "metrics": ["metric1", "metric2"]
                }}
            ],
            "education": [
                {{
            "institution": "University",
            "degree": "Degree",
            "field": "Field of study",
                    "duration": "Start - End",
                    "gpa": "GPA if mentioned"
                }}
            ],
    "skills": ["skill1", "skill2"],
            "certifications": ["cert1", "cert2"],
            "projects": ["project1", "project2"],
            "languages": ["language1", "language2"]
        }}
        
IMPORTANT:
- Extract ALL quantifiable metrics (numbers, percentages, dollar amounts)
- Use null for missing information
- Be thorough and accurate
- Return ONLY valid JSON

{validation_error}
""")
        
        for attempt in range(self.max_retries):
            try:
                validation_error = ""
                if attempt > 0:
                    validation_error = f"\nPREVIOUS ERROR: The last response had validation errors. Please ensure all fields match the required types and format."
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "cv_text": cv_text,
                    "validation_error": validation_error
                })
                
                # Extract JSON from response
                json_text = self._extract_json(response)
                parsed_data = json.loads(json_text)
                
                # Validate with Pydantic
                return CandidateProfileSchema(**parsed_data)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return minimal valid schema
                    return self._fallback_cv_parse(cv_text)
                continue
        
        return self._fallback_cv_parse(cv_text)
    
    def parse_job_description(self, jd_text: str) -> JobRequirementSchema:
        """
        Parse job description into structured JobRequirementSchema
        Implements retry mechanism for failed Pydantic validation
        """
        prompt = ChatPromptTemplate.from_template("""
You are an expert job description analyzer. Extract structured requirements from the job posting below.

Job Description:
{jd_text}

Extract ALL information and return as valid JSON matching this schema:
{{
    "title": "Job Title",
    "company": "Company Name",
    "location": "Location",
    "requirements": [
        {{
            "category": "Technical Skills|Experience|Education|Soft Skills",
            "requirement": "Specific requirement",
            "importance": "Required|Preferred|Nice to have",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "key_skills": ["skill1", "skill2"],
    "experience_level": "Entry|Mid|Senior|Executive",
    "industry": "Industry",
    "responsibilities": ["responsibility1", "responsibility2"],
    "salary_range": "Salary if mentioned",
    "benefits": ["benefit1", "benefit2"],
    "raw_text": "Full job description text"
}}

IMPORTANT:
- Extract ALL required and preferred skills
- Identify keywords for ATS optimization
- Categorize requirements clearly
- Return ONLY valid JSON

{validation_error}
""")
        
        for attempt in range(self.max_retries):
            try:
                validation_error = ""
                if attempt > 0:
                    validation_error = f"\nPREVIOUS ERROR: Please fix validation errors and return valid JSON."
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "jd_text": jd_text,
                    "validation_error": validation_error
                })
                
                # Extract JSON from response
                json_text = self._extract_json(response)
                parsed_data = json.loads(json_text)
                parsed_data["raw_text"] = jd_text
                
                # Validate with Pydantic
                return JobRequirementSchema(**parsed_data)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return self._fallback_jd_parse(jd_text)
                continue
        
        return self._fallback_jd_parse(jd_text)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        if isinstance(text, str):
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()
        return text
    
    def _fallback_cv_parse(self, cv_text: str) -> CandidateProfileSchema:
        """Fallback minimal CV parse"""
        from models import ContactInfo
        return CandidateProfileSchema(
            contact=ContactInfo(),
            summary="Unable to fully parse CV. Please review the original document.",
            experience=[],
            education=[],
            skills=[],
            certifications=[],
            projects=[],
            languages=[]
        )
    
    def _fallback_jd_parse(self, jd_text: str) -> JobRequirementSchema:
        """Fallback minimal JD parse"""
        return JobRequirementSchema(
            title="Position",
            requirements=[],
            key_skills=[],
            experience_level="Mid",
            responsibilities=[],
            raw_text=jd_text
        )


class ScoringAgent:
    """
    Scoring Agent: RAG-enhanced skill gap analysis and relevance scoring
    Uses Few-Shot Learning and explainable AI for transparency
    """
    
    def __init__(self, llm: Any, rag_system: RAGSystem):
        self.llm = llm
        self.rag = rag_system
    
    def score_match(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> MatchGapReportSchema:
        """
        Score CV-JD match using RAG-retrieved context
        Returns structured gap analysis and relevance score
        """
        # Generate multiple queries for RAG Fusion
        queries = [
            f"Skills and experience related to {job_requirements.get('title')}",
            f"Achievements relevant to {', '.join(job_requirements.get('key_skills', [])[:3])}",
            f"Experience in {job_requirements.get('industry', 'the industry')}"
        ]
        
        # Retrieve relevant context
        cv_context = self.rag.retrieve_multi_query(queries, collection="cv")
        jd_context = self.rag.retrieve_multi_query(queries, collection="jd")
        
        # Build prompt with Few-Shot examples
        prompt = ChatPromptTemplate.from_template("""
You are an expert career advisor analyzing CV-Job fit.

CANDIDATE FACTS (from RAG):
{cv_context}

JOB REQUIREMENTS (from RAG):
{jd_context}

CANDIDATE SKILLS:
{candidate_skills}

REQUIRED SKILLS:
{required_skills}

FEW-SHOT EXAMPLES:
Example 1 - High Match (Score: 90):
- Candidate has 8/10 required skills
- Strong relevant experience (5+ years)
- Minor gaps in emerging technologies
- Recommendation: "proceed"

Example 2 - Medium Match (Score: 65):
- Candidate has 5/10 required skills
- Some transferable experience
- Needs upskilling in key areas
- Recommendation: "optimize"

Example 3 - Low Match (Score: 40):
- Candidate has 3/10 required skills
- Limited relevant experience
- Major skill gaps
- Recommendation: "major_gaps"

NOW ANALYZE THIS CANDIDATE:

Provide analysis as JSON:
{{
    "relevance_score": 0-100,
    "skill_gaps": [
        {{
            "skill": "Missing skill name",
            "importance": "Required|Preferred",
            "present_in_cv": false,
            "suggested_evidence": "How to demonstrate this skill if present"
        }}
    ],
    "matched_skills": ["skill1", "skill2"],
    "target_keywords": ["keyword1", "keyword2"],
    "focus_areas": ["area1", "area2"],
    "recommendation": "proceed|optimize|major_gaps",
    "reasoning": "Explanation of score with citations to CV facts"
}}

SCORING CRITERIA:
- 90-100: Excellent match, proceed with rewriting
- 70-89: Good match, optimize certain areas
- 50-69: Moderate match, highlight transferable skills
- Below 50: Major gaps, may not be suitable

Return ONLY valid JSON.
""")
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "cv_context": "\n".join(cv_context) if cv_context else "No specific context retrieved",
                "jd_context": "\n".join(jd_context) if jd_context else "No specific context retrieved",
                "candidate_skills": ", ".join(candidate_profile.get("skills", [])),
                "required_skills": ", ".join(job_requirements.get("key_skills", []))
            })
            
            # Parse response
            json_text = self._extract_json(response)
            parsed_data = json.loads(json_text)
            
            return MatchGapReportSchema(**parsed_data)
                
        except Exception as e:
            # Fallback scoring
            return self._fallback_score(candidate_profile, job_requirements)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response"""
        if isinstance(text, str):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()
        return text
    
    def _fallback_score(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> MatchGapReportSchema:
        """Simple fallback scoring without LLM"""
        candidate_skills = set(candidate_profile.get("skills", []))
        required_skills = set(job_requirements.get("key_skills", []))
        
        matched = candidate_skills.intersection(required_skills)
        missing = required_skills.difference(candidate_skills)
        
        score = (len(matched) / len(required_skills) * 100) if required_skills else 50.0
        
        skill_gaps = [
            SkillGap(
                skill=skill,
                importance="Required",
                present_in_cv=False
            )
            for skill in missing
        ]
        
        return MatchGapReportSchema(
            relevance_score=score,
            skill_gaps=skill_gaps,
            matched_skills=list(matched),
            target_keywords=list(required_skills),
            focus_areas=["Skills", "Experience"],
            recommendation="optimize" if score >= 50 else "major_gaps",
            reasoning="Basic keyword matching analysis"
        )


class AnalysisAgent:
    """
    Analysis Agent: Generates concise CV analysis report organized by sections
    Provides specific, actionable recommendations for each CV section
    """
    
    def __init__(self, llm: Any, rag_system: RAGSystem):
        self.llm = llm
        self.rag = rag_system
    
    def analyze_cv(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any],
        match_report: Dict[str, Any]
    ) -> CVAnalysisReportSchema:
        """
        Analyze CV against job requirements and generate concise, actionable report
        Organized by sections with specific recommendations for each
        """
        # Retrieve relevant facts for grounding
        target_keywords = match_report.get("target_keywords", [])
        cv_facts = self.rag.retrieve_cv_facts(
            f"Experience and skills related to {', '.join(target_keywords[:5])}"
        )
        
        # Build comprehensive prompt for analysis
        prompt = ChatPromptTemplate.from_template("""
You are an expert career advisor analyzing a CV against job requirements.

CANDIDATE'S CURRENT CV:
- Summary: {candidate_summary}
- Experience: {candidate_experience}
- Skills: {candidate_skills}
- Education: {candidate_education}

JOB REQUIREMENTS:
- Title: {job_title}
- Required Skills: {required_skills}
- Key Responsibilities: {responsibilities}
- Experience Level: {experience_level}

MATCH ANALYSIS:
- Relevance Score: {relevance_score}%
- Matched Skills: {matched_skills}
- Skill Gaps: {skill_gaps}
- Target Keywords: {target_keywords}

Generate a CONCISE CV analysis report organized by sections. For each section, provide:
1. Current status assessment
2. Required changes (specific, actionable)
3. New points to add (with examples)
4. Keywords to integrate from job description

Return as JSON:
{{
    "overall_assessment": "Brief 2-3 sentence overall assessment of CV fit",
    "relevance_score": {relevance_score},
    "section_analyses": [
        {{
            "section_name": "Professional Summary",
            "current_status": "Brief assessment of current summary",
            "required_changes": ["Specific change 1", "Specific change 2"],
            "suggested_additions": ["New point to add with example", "Another addition"],
            "keywords_to_add": ["keyword1", "keyword2"],
            "priority": "high"
        }},
        {{
            "section_name": "Experience",
            "current_status": "Brief assessment of experience section",
            "required_changes": ["How to rephrase current experience", "What to emphasize"],
            "suggested_additions": ["New achievement bullet to add", "Another bullet focusing on X"],
            "keywords_to_add": ["keyword3", "keyword4"],
            "priority": "high"
        }},
        {{
            "section_name": "Skills",
            "current_status": "Brief assessment of skills section",
            "required_changes": ["Reorder to prioritize X", "Group by category"],
            "suggested_additions": ["Add skill1 if you have it", "Add skill2"],
            "keywords_to_add": ["skill_keyword1", "skill_keyword2"],
            "priority": "medium"
        }},
        {{
            "section_name": "Education & Certifications",
            "current_status": "Brief assessment",
            "required_changes": ["Specific changes needed"],
            "suggested_additions": ["Relevant certifications to highlight"],
            "keywords_to_add": ["education keywords"],
            "priority": "low"
        }}
    ],
    "critical_gaps": ["Must-have requirement 1 that's missing", "Critical gap 2"],
    "strengths_to_emphasize": ["Strength 1 already in CV", "Strength 2 to highlight more"],
    "quick_wins": ["Easy fix 1", "Quick improvement 2", "Low-hanging fruit 3"]
}}

INSTRUCTIONS:
- Be CONCISE and SPECIFIC
- Focus on ACTIONABLE recommendations
- Provide EXAMPLES of what to add
- Prioritize sections by importance (high/medium/low)
- Identify quick wins for immediate improvement
- Base all suggestions on actual job requirements
- Don't suggest fabricating experience

Return ONLY valid JSON.
""")
        
        try:
            # Format candidate data
            candidate_summary = candidate_profile.get("summary", "No summary provided")
            candidate_experience = self._format_experience(candidate_profile.get("experience", []))
            candidate_skills = ", ".join(candidate_profile.get("skills", []))
            candidate_education = self._format_education(candidate_profile.get("education", []))
            
            chain = prompt | self.llm
            response = chain.invoke({
                "candidate_summary": candidate_summary,
                "candidate_experience": candidate_experience,
                "candidate_skills": candidate_skills,
                "candidate_education": candidate_education,
                "job_title": job_requirements.get("title", "the position"),
                "required_skills": ", ".join(job_requirements.get("key_skills", [])),
                "responsibilities": ", ".join(job_requirements.get("responsibilities", [])[:5]),
                "experience_level": job_requirements.get("experience_level", "Not specified"),
                "relevance_score": match_report.get("relevance_score", 0),
                "matched_skills": ", ".join(match_report.get("matched_skills", [])),
                "skill_gaps": ", ".join([gap.get("skill", "") for gap in match_report.get("skill_gaps", [])[:5]]),
                "target_keywords": ", ".join(target_keywords[:15])
            })
            
            # Parse response
            json_text = self._extract_json(response)
            parsed_data = json.loads(json_text)
            
            return CVAnalysisReportSchema(**parsed_data)
                
        except Exception as e:
            return self._fallback_analysis(candidate_profile, job_requirements, match_report)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response"""
        if isinstance(text, str):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()
        return text
    
    def _format_experience(self, experience_list: List[Dict]) -> str:
        """Format experience for prompt"""
        formatted = []
        for exp in experience_list[:3]:  # Limit to top 3
            formatted.append(
                f"{exp.get('position')} at {exp.get('company')} ({exp.get('duration')}): "
                f"{exp.get('description', '')}"
            )
        return "\n".join(formatted) if formatted else "No experience provided"
    
    def _format_education(self, education_list: List[Dict]) -> str:
        """Format education for prompt"""
        formatted = []
        for edu in education_list:
            formatted.append(
                f"{edu.get('degree')} in {edu.get('field', 'N/A')} from {edu.get('institution')}"
            )
        return ", ".join(formatted) if formatted else "No education provided"
    
    def _fallback_analysis(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any],
        match_report: Dict[str, Any]
    ) -> CVAnalysisReportSchema:
        """Fallback if analysis fails"""
        return CVAnalysisReportSchema(
            overall_assessment="Unable to generate detailed analysis. Please review match report.",
            relevance_score=match_report.get("relevance_score", 0),
            section_analyses=[
                SectionAnalysis(
                    section_name="Skills",
                    current_status="Review needed",
                    required_changes=["Add missing skills from job requirements"],
                    suggested_additions=[f"Add {skill}" for skill in match_report.get("skill_gaps", [])[:3]],
                    keywords_to_add=match_report.get("target_keywords", [])[:5],
                    priority="high"
                )
            ],
            critical_gaps=[gap.get("skill", "") for gap in match_report.get("skill_gaps", [])[:3]],
            strengths_to_emphasize=match_report.get("matched_skills", [])[:3],
            quick_wins=["Update skills section", "Add keywords to summary"]
        )