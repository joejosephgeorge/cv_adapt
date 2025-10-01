"""
LangGraph Workflow for CV Adaptation
Implements multi-agent orchestration with cyclical refinement loop
"""
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from config import Config
from models import WorkflowState
from llm_factory import create_llm
from rag_system import RAGSystem
from agents import ParserAgent, ScoringAgent, RewriterAgent, QAAgent


class CVAdaptationWorkflow:
    """
    LangGraph workflow implementing the complete multi-agent CV adaptation pipeline
    with cyclical refinement for quality assurance
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize RAG system
        self.rag = RAGSystem(config.rag)
        
        # Create LLMs for each agent (hybrid strategy)
        parser_llm = create_llm(config.llm.get_provider_config("parser"))
        scoring_llm = create_llm(config.llm.get_provider_config("scoring"))
        rewriter_llm = create_llm(config.llm.get_provider_config("rewriter"))
        qa_llm = create_llm(config.llm.get_provider_config("qa"))
        
        # Initialize specialized agents
        self.parser = ParserAgent(parser_llm)
        self.scorer = ScoringAgent(scoring_llm, self.rag)
        self.rewriter = RewriterAgent(rewriter_llm, self.rag)
        self.qa = QAAgent(qa_llm, self.rag)
        
        # Build workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine with cyclical refinement"""
        # Use dict-based state instead of Pydantic model for LangGraph compatibility
        from typing import TypedDict
        
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("parse_documents", self._parse_documents)
        workflow.add_node("score_match", self._score_match)
        workflow.add_node("rewrite_cv", self._rewrite_cv)
        workflow.add_node("qa_validate", self._qa_validate)
        workflow.add_node("finalize", self._finalize)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("parse_documents")
        
        # Linear flow with conditional branching
        workflow.add_conditional_edges(
            "parse_documents",
            self._after_parsing,
            {
                "continue": "score_match",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "score_match",
            self._after_scoring,
            {
                "high_score": "finalize",  # Skip rewriting if score >= 95
                "proceed": "rewrite_cv",   # Score 70-94
                "optimize": "rewrite_cv",  # Score 50-69
                "fail": "finalize",        # Score < 50, don't waste resources
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "rewrite_cv",
            self._after_rewriting,
            {
                "continue": "qa_validate",
                "error": "handle_error"
            }
        )
        
        # Cyclical refinement loop: QA -> Rewrite or Finalize
        workflow.add_conditional_edges(
            "qa_validate",
            self._after_qa,
            {
                "pass": "finalize",
                "refine": "rewrite_cv",  # Loop back for self-correction
                "max_iterations": "finalize",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("finalize", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    # ========================================================================
    # Node Implementations
    # ========================================================================
    
    def _parse_documents(self, state: dict) -> dict:
        """Node A: Parse CV and JD into structured formats"""
        state["current_step"] = "parsing"
        
        try:
            # Parse CV
            if state.get("cv_text"):
                candidate_profile = self.parser.parse_cv(state["cv_text"])
                state["candidate_profile"] = candidate_profile
                
                # Index CV facts in RAG
                self.rag.index_cv_facts(candidate_profile.model_dump())
            else:
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append("No CV text provided")
                return state
            
            # Parse Job Description
            if state.get("job_description"):
                job_requirements = self.parser.parse_job_description(state["job_description"])
                state["job_requirements"] = job_requirements
                
                # Index job requirements in RAG
                self.rag.index_job_requirements(job_requirements.model_dump())
            else:
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append("No job description provided")
                return state
            
            state["current_step"] = "parsed"
            
        except Exception as e:
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Parsing error: {str(e)}")
            state["current_step"] = "error"
        
        return state
    
    def _score_match(self, state: dict) -> dict:
        """Node B: Score CV-JD match using RAG"""
        state["current_step"] = "scoring"
        
        try:
            if not state.get("candidate_profile") or not state.get("job_requirements"):
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append("Missing parsed data for scoring")
                return state
            
            # Score using RAG-enhanced agent
            match_report = self.scorer.score_match(
                state["candidate_profile"].model_dump(),
                state["job_requirements"].model_dump()
            )
            state["match_report"] = match_report
            
            state["current_step"] = "scored"
            
        except Exception as e:
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Scoring error: {str(e)}")
            state["current_step"] = "error"
        
        return state
    
    def _rewrite_cv(self, state: dict) -> dict:
        """Node C: Rewrite CV with RAG grounding (supports iterative refinement)"""
        state["current_step"] = "rewriting"
        
        try:
            if not all([state.get("candidate_profile"), state.get("job_requirements"), state.get("match_report")]):
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append("Missing data for rewriting")
                return state
            
            # Get QA feedback if this is a refinement iteration
            qa_feedback = None
            qa_report = state.get("qa_report")
            if qa_report and not qa_report.passed:
                qa_feedback = qa_report.feedback_for_rewrite
            
            # Rewrite using RAG-grounded agent
            rewritten_sections = self.rewriter.rewrite_cv(
                state["candidate_profile"].model_dump(),
                state["job_requirements"].model_dump(),
                state["match_report"].model_dump(),
                qa_feedback=qa_feedback
            )
            state["rewritten_sections"] = rewritten_sections
            
            state["current_step"] = "rewritten"
            
        except Exception as e:
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Rewriting error: {str(e)}")
            state["current_step"] = "error"
        
        return state
    
    def _qa_validate(self, state: dict) -> dict:
        """Node D: QA validation with Self-RAG"""
        state["current_step"] = "qa_validation"
        
        try:
            if not state.get("rewritten_sections"):
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append("No rewritten content to validate")
                return state
            
            # Get original CV facts for verification
            original_facts = state.get("cv_rag_context", [])
            
            # Validate using QA agent
            match_report = state.get("match_report")
            job_requirements = state.get("job_requirements")
            
            qa_report = self.qa.validate_cv(
                state["rewritten_sections"].model_dump(),
                original_facts,
                match_report.target_keywords if match_report else [],
                job_requirements.model_dump() if job_requirements else {}
            )
            state["qa_report"] = qa_report
            
            state["qa_iteration_count"] = state.get("qa_iteration_count", 0) + 1
            state["current_step"] = "qa_completed"
            
        except Exception as e:
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"QA validation error: {str(e)}")
            state["current_step"] = "error"
        
        return state
    
    def _finalize(self, state: dict) -> dict:
        """Node E: Finalize and prepare output"""
        state["current_step"] = "finalizing"
        
        try:
            from models import AdaptedCVSchema
            
            rewritten_sections = state.get("rewritten_sections")
            candidate_profile = state.get("candidate_profile")
            match_report = state.get("match_report")
            qa_report = state.get("qa_report")
            
            # Compile final adapted CV
            if rewritten_sections and candidate_profile:
                adapted_cv = AdaptedCVSchema(
                    contact=candidate_profile.contact,
                    summary=rewritten_sections.summary or candidate_profile.summary or "",
                    experience=rewritten_sections.experience,
                    education=candidate_profile.education,
                    skills=rewritten_sections.skills,
                    certifications=candidate_profile.certifications,
                    projects=candidate_profile.projects,
                    languages=candidate_profile.languages,
                    relevance_score=match_report.relevance_score if match_report else 0.0,
                    qa_passed=qa_report.passed if qa_report else False,
                    adaptation_notes=rewritten_sections.modifications_made if rewritten_sections else []
                )
                state["adapted_cv"] = adapted_cv
            elif candidate_profile:
                # No rewriting performed (high initial score or low score)
                adapted_cv = AdaptedCVSchema(
                    contact=candidate_profile.contact,
                    summary=candidate_profile.summary or "",
                    experience=candidate_profile.experience,
                    education=candidate_profile.education,
                    skills=candidate_profile.skills,
                    certifications=candidate_profile.certifications,
                    projects=candidate_profile.projects,
                    languages=candidate_profile.languages,
                    relevance_score=match_report.relevance_score if match_report else 0.0,
                    qa_passed=True,
                    adaptation_notes=["Original CV returned without modifications"]
                )
                state["adapted_cv"] = adapted_cv
            
            state["current_step"] = "completed"
            state["should_continue"] = False
            
        except Exception as e:
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Finalization error: {str(e)}")
            state["current_step"] = "error"
        
        return state
    
    def _handle_error(self, state: dict) -> dict:
        """Error handler node"""
        state["current_step"] = "error"
        state["should_continue"] = False
        return state
    
    # ========================================================================
    # Conditional Edge Functions
    # ========================================================================
    
    def _after_parsing(self, state: dict) -> Literal["continue", "error"]:
        """Decide flow after parsing"""
        if state.get("errors"):
            return "error"
        return "continue"
    
    def _after_scoring(
        self,
        state: dict
    ) -> Literal["high_score", "proceed", "optimize", "fail", "error"]:
        """Decide flow after scoring (conditional branching based on score)"""
        if state.get("errors"):
            return "error"
        
        match_report = state.get("match_report")
        if not match_report:
            return "error"
        
        score = match_report.relevance_score
        
        if score >= 95:
            return "high_score"  # Perfect match, skip rewriting
        elif score >= 70:
            return "proceed"  # Good match, optimize
        elif score >= 50:
            return "optimize"  # Moderate match, emphasize transferable skills
        else:
            return "fail"  # Poor match, don't waste resources
    
    def _after_rewriting(self, state: dict) -> Literal["continue", "error"]:
        """Decide flow after rewriting"""
        if state.get("errors"):
            return "error"
        return "continue"
    
    def _after_qa(
        self,
        state: dict
    ) -> Literal["pass", "refine", "max_iterations", "error"]:
        """Decide flow after QA (cyclical refinement logic)"""
        if state.get("errors"):
            return "error"
        
        qa_report = state.get("qa_report")
        if not qa_report:
            return "error"
        
        # Check if max iterations reached
        max_iterations = self.config.workflow.max_qa_iterations
        qa_iteration_count = state.get("qa_iteration_count", 0)
        if qa_iteration_count >= max_iterations:
            return "max_iterations"
        
        # Check if QA passed
        if qa_report.passed:
            return "pass"
        
        # Check if refinement is enabled
        if not self.config.workflow.enable_self_correction:
            return "pass"  # Skip refinement, proceed with current version
        
        # Refine (loop back to rewriting)
        return "refine"
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def run(
        self,
        cv_text: str,
        job_description: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run the complete CV adaptation workflow
        
        Args:
            cv_text: Raw CV text
            job_description: Raw job description text
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary with results
        """
        # Initialize state as dict for LangGraph
        initial_state = {
            "cv_text": cv_text,
            "job_description": job_description,
            "errors": [],
            "qa_iteration_count": 0,
            "current_step": "start",
            "should_continue": True,
            "cv_rag_context": [],
            "jd_rag_context": []
        }
        
        # Run workflow with progress tracking
        if progress_callback:
            progress_callback("Starting CV adaptation workflow...")
        
        result_state = self.graph.invoke(initial_state)
        
        # LangGraph returns state as dict, not Pydantic object
        if isinstance(result_state, dict):
            errors = result_state.get("errors", [])
            adapted_cv = result_state.get("adapted_cv")
            match_report = result_state.get("match_report")
            qa_report = result_state.get("qa_report")
            qa_iterations = result_state.get("qa_iteration_count", 0)
            current_step = result_state.get("current_step", "unknown")
        else:
            # Fallback if it's a Pydantic object
            errors = result_state.errors
            adapted_cv = result_state.adapted_cv
            match_report = result_state.match_report
            qa_report = result_state.qa_report
            qa_iterations = result_state.qa_iteration_count
            current_step = result_state.current_step
        
        # Extract results
        return {
            "success": len(errors) == 0 and adapted_cv is not None,
            "adapted_cv": adapted_cv,
            "match_report": match_report,
            "qa_report": qa_report,
            "qa_iterations": qa_iterations,
            "errors": errors,
            "current_step": current_step
        }