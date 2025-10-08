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
You are an expert ATS-optimization career advisor and CV writer analyzing a CV against job requirements.

CANDIDATE'S CURRENT CV:
- Summary: {candidate_summary}
- Experience: {candidate_experience}
- Skills: {candidate_skills}
- Education: {candidate_education}
- Certifications: {candidate_certifications}
- Projects: {candidate_projects}

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

Generate an EXTREMELY DETAILED, ATS-OPTIMIZED CV analysis report organized by sections. 

For each CV section, provide GRANULAR, SPECIFIC, and ACTIONABLE recommendations:
1. Current status assessment (exactly what's in the CV now)
2. Items to ADD (with specific examples of rewritten content, frameworks, tools to mention)
3. Items to REMOVE (identify specific weak/outdated phrases or content)
4. Items to MODIFY (provide BEFORE → AFTER rewrite examples)
5. ATS Keywords to integrate (extract from job description)

Return as JSON:
{{
    "overall_assessment": "Detailed 2-3 sentence assessment covering CV-job fit, ATS readiness, and overall alignment",
    "relevance_score": {relevance_score},
    "section_analyses": [
        {{
            "section_name": "Professional Summary",
            "current_status": "Currently: Missing/Present - describe exactly what exists",
            "items_to_add": [
                "Add 3-4 line professional summary highlighting [specific skills from JD]",
                "Include years of experience and key domain expertise (e.g., 'ML Engineer with 5+ years in predictive modeling')",
                "Mention specific tools/frameworks from CV that match JD (e.g., 'Skilled in Python, TensorFlow, Scikit-learn')",
                "Add business impact focus (e.g., 'driving data-driven decision making')"
            ],
            "items_to_remove": ["Generic phrases like 'hard worker' or 'team player'", "Vague statements without metrics"],
            "items_to_modify": ["If summary exists: BEFORE: '[current text]' → AFTER: '[rewritten with keywords and metrics]'"],
            "keywords_to_add": ["[exact keywords from job description]", "[ATS-friendly terms]"],
            "priority": "high"
        }},
        {{
            "section_name": "Experience - [Company Name]",
            "current_status": "Currently has [X] bullets covering [topics]. Strong on metrics but missing [specific JD requirement]",
            "items_to_add": [
                "Add bullet: 'Implemented [specific ML algorithms from JD, e.g., XGBoost, Random Forest] to [business outcome]'",
                "Add collaboration/stakeholder element: 'Created Tableau dashboards to visualize [metrics] for [stakeholders]'",
                "Add brainstorming/innovation mention: 'Participated in cross-functional brainstorming sessions to identify AI-driven improvements'",
                "Add specific tools/frameworks mentioned in JD but missing from bullets"
            ],
            "items_to_remove": ["Routine tasks without business impact", "Bullets lacking metrics or outcomes"],
            "items_to_modify": [
                "BEFORE: 'Deployed a recommendation engine' → AFTER: 'Deployed a collaborative filtering recommendation engine using Python (Scikit-learn) to increase order size by 7%'",
                "BEFORE: 'Implemented forecasting' → AFTER: 'Implemented time series forecasting algorithms (ARIMA, Prophet) to predict surge demand'",
                "Add specific algorithm names, tools, and frameworks to existing bullets"
            ],
            "keywords_to_add": ["machine learning algorithms", "predictive modeling", "data visualization", "stakeholder communication"],
            "priority": "high"
        }},
        {{
            "section_name": "Skills",
            "current_status": "Current skills: [list them]. Missing: [JD requirements not present]",
            "items_to_add": [
                "Add 'Machine Learning Implementation' or 'ML Algorithm Development' as category",
                "Add 'Data Visualization' with tools: Tableau, Power BI, Plotly, Matplotlib",
                "Add 'Predictive Modeling' explicitly",
                "Add 'Business Process Optimization' or 'Data-Driven Optimization'",
                "Add 'Dashboard Development' or 'KPI Reporting'",
                "Add specific missing tools from JD: [list them]"
            ],
            "items_to_remove": ["Outdated technologies not relevant to role", "Basic skills assumed for level (e.g., Excel for senior roles)"],
            "items_to_modify": [
                "Reorder: Put JD-matching skills first",
                "Group by category: 'ML & AI', 'Data Analysis & Visualization', 'Tools & Technologies', 'Business Skills'",
                "Expand abbreviated terms for ATS (e.g., 'ML' → 'Machine Learning (ML)')"
            ],
            "keywords_to_add": ["exact skill keywords from job description"],
            "priority": "high"
        }},
        {{
            "section_name": "Education & Certifications",
            "current_status": "Currently has [degree]. Missing certifications/training",
            "items_to_add": [
                "Add relevant certifications: 'Machine Learning (Coursera/Stanford)', 'Data Visualization (Tableau/Power BI)'",
                "Add relevant coursework if recent graduate: 'Relevant coursework: [list matching JD]'",
                "Add online courses showing continuous learning commitment"
            ],
            "items_to_remove": ["High school education if college degree present", "Very old/irrelevant certifications"],
            "items_to_modify": ["Highlight honors, relevant GPA if strong (>3.5), relevant thesis/capstone projects"],
            "keywords_to_add": ["certification names from JD", "continuous learning", "professional development"],
            "priority": "medium"
        }},
        {{
            "section_name": "Projects",
            "current_status": "Missing projects section / Has [X] projects but [issue]",
            "items_to_add": [
                "Add 'Projects' section if missing",
                "Add 2-3 projects: '[Project Name]: Brief description with tech stack and business impact'",
                "Example: 'Predictive Customer Churn Model: Built Random Forest classifier achieving 85% accuracy, deployed via Flask API'",
                "Focus on projects matching JD requirements"
            ],
            "items_to_remove": ["Very old projects with outdated tech", "Projects not relevant to target role"],
            "items_to_modify": ["Add metrics and business outcomes to existing project descriptions", "Mention specific tools and algorithms used"],
            "keywords_to_add": ["project-related keywords from JD"],
            "priority": "medium"
        }}
    ],
    "critical_gaps": ["Specific must-have skills/experience missing from CV", "ATS keyword gaps"],
    "strengths_to_emphasize": ["Strong metrics and quantifiable achievements", "Relevant experience with [specific skill]"],
    "quick_wins": ["Add professional summary", "Insert ATS keywords in skills section", "Mention specific algorithms/tools in experience bullets"]
}}

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

1. **ATS OPTIMIZATION FOCUS:**
   - Extract EXACT keywords and phrases from job description
   - Suggest where to insert these keywords naturally
   - Recommend expanding abbreviations for ATS scanning (ML → Machine Learning)
   - Focus on making CV ATS-friendly while remaining readable

2. **BE EXTREMELY GRANULAR AND SPECIFIC:**
   - Don't just say "add skills" - specify WHICH skills, tools, frameworks
   - Don't just say "improve bullets" - provide EXACT BEFORE → AFTER rewrites
   - Reference specific bullet points from candidate's CV by quoting them
   - Provide actual rewritten examples the candidate can copy/paste
   - Mention specific algorithm names, tool versions, framework names

3. **FOR ADDITIONS:**
   - Provide concrete, copy-paste ready examples
   - Base additions on candidate's EXISTING experience (don't fabricate)
   - Show how to reframe existing work to match JD requirements
   - Suggest specific metrics, tools, or frameworks to mention
   - Example: "Add: 'Implemented supervised learning models (Random Forest, XGBoost) using Scikit-learn and Python'"

4. **FOR REMOVALS:**
   - Quote SPECIFIC phrases or bullets to remove
   - Explain WHY (outdated, generic, weak, irrelevant)
   - Example: "Remove: 'Responsible for data analysis' - too vague, lacks impact"

5. **FOR MODIFICATIONS:**
   - Provide BEFORE → AFTER examples for every suggestion
   - Show exactly how to rewrite bullets to add keywords, metrics, specificity
   - Example: "BEFORE: 'Built a model' → AFTER: 'Developed gradient boosting classification model (XGBoost) achieving 92% accuracy'"
   - Focus on adding: algorithm names, tool names, metrics, business impact, stakeholder mentions

6. **SECTION-SPECIFIC GUIDANCE:**
   - **Summary**: Write a complete 3-4 line example summary if missing
   - **Experience**: Provide 2-3 complete rewritten bullet examples per role
   - **Skills**: List 10-15 specific skills/tools to add from JD
   - **Education**: Suggest specific certifications or courses matching the role
   - **Projects**: Provide project description templates

7. **KEYWORD INTEGRATION:**
   - Extract 20-30 keywords from job description
   - Show EXACTLY where to insert each keyword category
   - Group keywords by: technical skills, soft skills, domain terms, tools

8. **QUALITY STANDARDS:**
   - Every recommendation must be actionable and specific
   - Provide examples the candidate can immediately use
   - Don't suggest fabricating experience
   - Base all suggestions on candidate's actual background
   - Analyze ALL CV sections: Summary, each Experience entry, Skills, Education, Certifications, Projects

Return ONLY valid JSON with these detailed, specific, actionable recommendations.
""")
        
        try:
            # Format candidate data
            candidate_summary = candidate_profile.get("summary", "No summary provided")
            candidate_experience = self._format_experience(candidate_profile.get("experience", []))
            candidate_skills = ", ".join(candidate_profile.get("skills", [])) if candidate_profile.get("skills") else "Not specified"
            candidate_education = self._format_education(candidate_profile.get("education", []))
            candidate_certifications = ", ".join(candidate_profile.get("certifications", [])) if candidate_profile.get("certifications") else "None listed"
            candidate_projects = "; ".join(candidate_profile.get("projects", [])[:5]) if candidate_profile.get("projects") else "No projects listed"
            
            # Format skill gaps with importance
            skill_gaps_formatted = []
            for gap in match_report.get("skill_gaps", [])[:10]:
                if isinstance(gap, dict):
                    skill_gaps_formatted.append(f"{gap.get('skill', '')} ({gap.get('importance', 'unknown')})")
                else:
                    skill_gaps_formatted.append(str(gap))
            
            chain = prompt | self.llm
            response = chain.invoke({
                "candidate_summary": candidate_summary,
                "candidate_experience": candidate_experience,
                "candidate_skills": candidate_skills,
                "candidate_education": candidate_education,
                "candidate_certifications": candidate_certifications,
                "candidate_projects": candidate_projects,
                "job_title": job_requirements.get("title", "the position"),
                "required_skills": ", ".join(job_requirements.get("key_skills", [])[:20]),
                "responsibilities": "; ".join(job_requirements.get("responsibilities", [])[:10]),
                "experience_level": job_requirements.get("experience_level", "Not specified"),
                "relevance_score": match_report.get("relevance_score", 0),
                "matched_skills": ", ".join(match_report.get("matched_skills", [])[:15]),
                "skill_gaps": ", ".join(skill_gaps_formatted),
                "target_keywords": ", ".join(target_keywords[:30])
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
        """Format experience for prompt with full details"""
        formatted = []
        for exp in experience_list[:5]:  # Include more roles for better analysis
            exp_text = f"\n{exp.get('position')} at {exp.get('company')} ({exp.get('duration')})"
            if exp.get('description'):
                exp_text += f"\n  Description: {exp.get('description')}"
            
            if exp.get('achievements'):
                exp_text += f"\n  Achievements:"
                for achievement in exp.get('achievements', [])[:5]:
                    exp_text += f"\n    • {achievement}"
            
            if exp.get('skills_used'):
                exp_text += f"\n  Skills Used: {', '.join(exp.get('skills_used', []))}"
            
            if exp.get('metrics'):
                exp_text += f"\n  Metrics: {', '.join(exp.get('metrics', []))}"
            
            formatted.append(exp_text)
        return "\n".join(formatted) if formatted else "No experience provided"
    
    def _format_education(self, education_list: List[Dict]) -> str:
        """Format education for prompt with full details"""
        formatted = []
        for edu in education_list:
            edu_text = f"{edu.get('degree')} in {edu.get('field', 'N/A')} from {edu.get('institution')} ({edu.get('duration', 'N/A')})"
            if edu.get('gpa'):
                edu_text += f" - GPA: {edu.get('gpa')}"
            formatted.append(edu_text)
        return "\n".join(formatted) if formatted else "No education provided"
    
    def _fallback_analysis(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any],
        match_report: Dict[str, Any]
    ) -> CVAnalysisReportSchema:
        """Fallback if analysis fails"""
        skill_gaps = match_report.get("skill_gaps", [])
        return CVAnalysisReportSchema(
            overall_assessment="Unable to generate detailed analysis. Please review match report.",
            relevance_score=match_report.get("relevance_score", 0),
            section_analyses=[
                SectionAnalysis(
                    section_name="Skills",
                    current_status="Review needed - basic analysis only",
                    items_to_add=[f"Add {gap.get('skill', skill) if isinstance(gap, dict) else skill}" 
                                 for skill in (skill_gaps if isinstance(skill_gaps, list) else [])[:3]
                                 for gap in [skill if isinstance(skill, dict) else {'skill': skill}]],
                    items_to_remove=["Review skills section for outdated or irrelevant skills"],
                    items_to_modify=["Reorder skills to match job requirements priority"],
                    keywords_to_add=match_report.get("target_keywords", [])[:5],
                    priority="high"
                )
            ],
            critical_gaps=[gap.get("skill", str(gap)) if isinstance(gap, dict) else str(gap) 
                          for gap in skill_gaps[:3]],
            strengths_to_emphasize=match_report.get("matched_skills", [])[:3],
            quick_wins=["Update skills section", "Add keywords to summary", "Reorder experience bullets"]
        )