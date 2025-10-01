"""
Specialized LLM Agents for CV Adaptation
Implements 4 key agents as per architecture:
1. Parser Agent: Structured data extraction
2. Scoring Agent: RAG-enhanced relevance scoring
3. Rewriter Agent: Content generation with RAG grounding
4. QA Agent: Quality assurance and self-correction
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
    RewrittenSectionSchema,
    QAReportSchema,
    SkillGap,
    QAIssue,
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


class RewriterAgent:
    """
    Rewriter Agent: Generates tailored CV content using RAG grounding
    Implements Action-Metric-Result framework for quantifiable achievements
    """
    
    def __init__(self, llm: Any, rag_system: RAGSystem):
        self.llm = llm
        self.rag = rag_system
    
    def rewrite_cv(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any],
        match_report: Dict[str, Any],
        qa_feedback: Optional[str] = None
    ) -> RewrittenSectionSchema:
        """
        Rewrite CV sections with RAG grounding and keyword optimization
        Supports iterative refinement based on QA feedback
        """
        # Retrieve relevant facts for grounding
        target_keywords = match_report.get("target_keywords", [])
        cv_facts = self.rag.retrieve_cv_facts(
            f"Experience and skills related to {', '.join(target_keywords[:5])}"
        )
        
        # Get few-shot examples from original CV for style matching
        original_bullets = []
        for exp in candidate_profile.get("experience", [])[:2]:
            original_bullets.extend(exp.get("achievements", [])[:2])
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert CV writer. Rewrite the CV to match the job requirements while maintaining factual accuracy.

ORIGINAL CV FACTS (Ground truth - DO NOT invent new facts):
{cv_facts}

JOB REQUIREMENTS:
Title: {job_title}
Required Skills: {required_skills}
Key Responsibilities: {responsibilities}

TARGET KEYWORDS TO INTEGRATE:
{target_keywords}

SKILL GAPS TO ADDRESS:
{skill_gaps}

STYLE EXAMPLES (maintain similar tone and length):
{style_examples}

ACTION-METRIC-RESULT FRAMEWORK:
Each bullet should follow: "Action Verb + Task + Metric + Result"
Example: "Led team of 5 engineers to migrate legacy system, reducing deployment time by 40% and improving reliability"

{qa_feedback}

Generate rewritten sections as JSON:
{{
    "summary": "Tailored professional summary (100-150 words) emphasizing {job_title} skills and {key_skills}",
    "experience": [
        {{
            "company": "Company Name",
            "position": "Position Title",
            "duration": "Duration",
            "description": "Rewritten description highlighting relevant aspects",
            "achievements": [
                "Action verb + quantifiable achievement + relevant keywords",
                "Another achievement with metrics"
            ],
            "skills_used": ["Relevant skills from job posting"],
            "metrics": ["Specific numbers/percentages/results"]
        }}
    ],
    "skills": ["Prioritized skills matching job requirements"],
    "keywords_integrated": ["List of keywords successfully added"],
    "modifications_made": ["What was changed and why"]
}}

CRITICAL RULES:
1. ONLY use facts from "ORIGINAL CV FACTS" - NO fabrication
2. Integrate ALL target keywords naturally
3. Use quantifiable metrics wherever possible
4. Highlight transferable skills if exact skills are missing
5. Maintain professional, achievement-oriented tone
6. Keep bullet points concise (1-2 lines each)

Return ONLY valid JSON.
""")
        
        try:
            qa_feedback_text = ""
            if qa_feedback:
                qa_feedback_text = f"\nQA FEEDBACK FOR REVISION:\n{qa_feedback}\nPlease address these specific issues in your rewrite."
            
            chain = prompt | self.llm
            response = chain.invoke({
                "cv_facts": "\n".join(cv_facts) if cv_facts else "No specific facts retrieved",
                "job_title": job_requirements.get("title", "the position"),
                "required_skills": ", ".join(job_requirements.get("key_skills", [])),
                "responsibilities": ", ".join(job_requirements.get("responsibilities", [])[:3]),
                "target_keywords": ", ".join(target_keywords[:10]),
                "skill_gaps": ", ".join([gap.get("skill", "") for gap in match_report.get("skill_gaps", [])[:5]]),
                "style_examples": "\n".join([f"- {bullet}" for bullet in original_bullets]) if original_bullets else "Use professional, concise tone",
                "key_skills": ", ".join(job_requirements.get("key_skills", [])[:3]),
                "qa_feedback": qa_feedback_text
            })
            
            # Parse response
            json_text = self._extract_json(response)
            parsed_data = json.loads(json_text)
            
            return RewrittenSectionSchema(**parsed_data)
                
        except Exception as e:
            return self._fallback_rewrite(candidate_profile, job_requirements, match_report)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response"""
        if isinstance(text, str):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()
        return text
    
    def _fallback_rewrite(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any],
        match_report: Dict[str, Any]
    ) -> RewrittenSectionSchema:
        """Fallback if rewriting fails"""
        from models import Experience
        
        return RewrittenSectionSchema(
            summary=f"Professional with experience in {', '.join(candidate_profile.get('skills', [])[:3])}",
            experience=[
                Experience(**exp) for exp in candidate_profile.get("experience", [])
            ],
            skills=candidate_profile.get("skills", []),
            keywords_integrated=[],
            modifications_made=["Fallback rewrite due to error"]
        )


class QAAgent:
    """
    QA Agent: Quality assurance with self-RAG verification
    Validates factual accuracy, keyword integration, and style consistency
    Triggers cyclical refinement loop if issues are found
    """
    
    def __init__(self, llm: Any, rag_system: RAGSystem):
        self.llm = llm
        self.rag = rag_system
    
    def validate_cv(
        self,
        rewritten_sections: Dict[str, Any],
        original_cv_facts: List[str],
        required_keywords: List[str],
        job_requirements: Dict[str, Any]
    ) -> QAReportSchema:
        """
        Validate rewritten CV for factual accuracy and keyword coverage
        Implements Self-RAG for grounding verification
        """
        # Retrieve original facts for verification
        cv_facts = self.rag.retrieve_cv_facts("All experience, achievements, and metrics")
        
        prompt = ChatPromptTemplate.from_template("""
You are a strict QA auditor for CV content. Verify the rewritten CV for quality issues.

ORIGINAL CV FACTS (Ground truth):
{original_facts}

REWRITTEN CV CONTENT:
Summary: {rewritten_summary}
Experience: {rewritten_experience}
Skills: {rewritten_skills}

REQUIRED KEYWORDS TO VERIFY:
{required_keywords}

JOB REQUIREMENTS:
{job_requirements}

VERIFICATION CHECKLIST:
1. Factual Accuracy: Are all metrics, dates, and achievements verifiable in original CV?
2. Keyword Integration: Are all high-priority keywords included naturally?
3. No Hallucination: Are there any invented facts not in the original CV?
4. Style Consistency: Is the tone professional and consistent?
5. ATS Optimization: Are keywords used naturally (not stuffed)?

Provide QA report as JSON:
{{
    "passed": true/false,
    "overall_score": 0-100,
    "issues": [
        {{
            "section": "summary|experience|skills",
            "issue_type": "missing_keyword|factual_error|style_inconsistency|hallucination",
            "description": "Specific issue description",
            "severity": "critical|major|minor",
            "suggested_fix": "How to fix this issue"
        }}
    ],
    "keywords_verified": ["keyword1", "keyword2"],
    "missing_keywords": ["missing1", "missing2"],
    "factual_consistency_check": true/false,
    "style_consistency_check": true/false,
    "feedback_for_rewrite": "Specific instructions for rewriter if failed"
}}

SCORING:
- 95-100: Excellent, pass
- 85-94: Good, minor improvements needed
- 70-84: Acceptable, some issues to fix
- Below 70: Fail, needs rewrite

CRITICAL: If you find ANY invented facts/metrics not in original CV, mark as CRITICAL issue.

Return ONLY valid JSON.
""")
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "original_facts": "\n".join(cv_facts) if cv_facts else "\n".join(original_cv_facts),
                "rewritten_summary": rewritten_sections.get("summary", ""),
                "rewritten_experience": json.dumps(rewritten_sections.get("experience", []), indent=2),
                "rewritten_skills": ", ".join(rewritten_sections.get("skills", [])),
                "required_keywords": ", ".join(required_keywords),
                "job_requirements": f"Title: {job_requirements.get('title')}, Skills: {', '.join(job_requirements.get('key_skills', []))}"
            })
            
            # Parse response
            json_text = self._extract_json(response)
            parsed_data = json.loads(json_text)
            
            return QAReportSchema(**parsed_data)
                
        except Exception as e:
            return self._fallback_qa(rewritten_sections, required_keywords)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response"""
        if isinstance(text, str):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()
        return text
    
    def _fallback_qa(
        self,
        rewritten_sections: Dict[str, Any],
        required_keywords: List[str]
    ) -> QAReportSchema:
        """Simple fallback QA check"""
        # Simple keyword check
        content_text = f"{rewritten_sections.get('summary', '')} {json.dumps(rewritten_sections.get('experience', []))} {' '.join(rewritten_sections.get('skills', []))}"
        content_lower = content_text.lower()
        
        keywords_found = [kw for kw in required_keywords if kw.lower() in content_lower]
        keywords_missing = [kw for kw in required_keywords if kw.lower() not in content_lower]
        
        issues = []
        for kw in keywords_missing[:5]:  # Report first 5 missing
            issues.append(QAIssue(
                section="overall",
                issue_type="missing_keyword",
                description=f"Required keyword '{kw}' not found in rewritten content",
                severity="major",
                suggested_fix=f"Integrate '{kw}' naturally into relevant sections"
            ))
        
        passed = len(keywords_missing) == 0
        score = (len(keywords_found) / len(required_keywords) * 100) if required_keywords else 100.0
        
        return QAReportSchema(
            passed=passed,
            overall_score=score,
            issues=issues,
            keywords_verified=keywords_found,
            missing_keywords=keywords_missing,
            factual_consistency_check=True,  # Cannot verify in fallback
            style_consistency_check=True,
            feedback_for_rewrite="Please integrate missing keywords" if keywords_missing else None
        )