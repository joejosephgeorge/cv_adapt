"""
LangGraph Workflow for CV Analysis
Implements multi-agent orchestration for CV analysis report generation
"""
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from config import Config
from models import WorkflowState
from llm_factory import create_llm
from rag_system import RAGSystem
from agents import ParserAgent, ScoringAgent, AnalysisAgent


class CVAdaptationWorkflow:
    """
    LangGraph workflow implementing CV analysis report generation
    Streamlined pipeline without QA loops
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize RAG system
        self.rag = RAGSystem(config.rag)
        
        # Create LLMs for each agent (hybrid strategy)
        parser_llm = create_llm(config.llm.get_provider_config("parser"))
        scoring_llm = create_llm(config.llm.get_provider_config("scoring"))
        analysis_llm = create_llm(config.llm.get_provider_config("rewriter"))  # Use rewriter config for analysis
        
        # Initialize specialized agents
        self.parser = ParserAgent(parser_llm)
        self.scorer = ScoringAgent(scoring_llm, self.rag)
        self.analyzer = AnalysisAgent(analysis_llm, self.rag)
        
        # Build workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for CV analysis"""
        # Use dict-based state instead of Pydantic model for LangGraph compatibility
        workflow = StateGraph(dict)
        
        # Add nodes - simplified pipeline
        workflow.add_node("parse_documents", self._parse_documents)
        workflow.add_node("score_match", self._score_match)
        workflow.add_node("analyze_cv", self._analyze_cv)
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
                "continue": "analyze_cv",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_cv",
            self._after_analysis,
            {
                "continue": "finalize",
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
    
    def _analyze_cv(self, state: dict) -> dict:
        """Node C: Analyze CV and generate concise report"""
        state["current_step"] = "analyzing"
        
        try:
            if not all([state.get("candidate_profile"), state.get("job_requirements"), state.get("match_report")]):
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append("Missing data for analysis")
                return state
            
            # Generate analysis using AnalysisAgent
            analysis_report = self.analyzer.analyze_cv(
                state["candidate_profile"].model_dump(),
                state["job_requirements"].model_dump(),
                state["match_report"].model_dump()
            )
            state["analysis_report"] = analysis_report
            
            state["current_step"] = "analyzed"
            
        except Exception as e:
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Analysis error: {str(e)}")
            state["current_step"] = "error"
        
        return state
    
    
    def _finalize(self, state: dict) -> dict:
        """Node D: Finalize and prepare analysis output"""
        state["current_step"] = "finalizing"
        
        try:
            from models import CVAnalysisOutputSchema
            
            analysis_report = state.get("analysis_report")
            candidate_profile = state.get("candidate_profile")
            match_report = state.get("match_report")
            job_requirements = state.get("job_requirements")
            
            # Compile final analysis output
            if analysis_report and match_report:
                final_output = CVAnalysisOutputSchema(
                    analysis_report=analysis_report,
                    match_report=match_report,
                    candidate_name=candidate_profile.contact.name if candidate_profile and candidate_profile.contact else None,
                    job_title=job_requirements.title if job_requirements else None
                )
                state["final_output"] = final_output
            
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
    ) -> Literal["continue", "error"]:
        """Decide flow after scoring"""
        if state.get("errors"):
            return "error"
        
        match_report = state.get("match_report")
        if not match_report:
            return "error"
        
        return "continue"  # Always proceed to analysis
    
    def _after_analysis(self, state: dict) -> Literal["continue", "error"]:
        """Decide flow after analysis"""
        if state.get("errors"):
            return "error"
        return "continue"
    
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
        Run the complete CV analysis workflow
        
        Args:
            cv_text: Raw CV text
            job_description: Raw job description text
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary with results including analysis report
        """
        # Initialize state as dict for LangGraph
        initial_state = {
            "cv_text": cv_text,
            "job_description": job_description,
            "errors": [],
            "current_step": "start",
            "should_continue": True,
            "cv_rag_context": [],
            "jd_rag_context": []
        }
        
        # Run workflow with progress tracking
        if progress_callback:
            progress_callback("Starting CV analysis workflow...")
        
        result_state = self.graph.invoke(initial_state)
        
        # LangGraph returns state as dict, not Pydantic object
        if isinstance(result_state, dict):
            errors = result_state.get("errors", [])
            final_output = result_state.get("final_output")
            analysis_report = result_state.get("analysis_report")
            match_report = result_state.get("match_report")
            current_step = result_state.get("current_step", "unknown")
        else:
            # Fallback if it's a Pydantic object
            errors = result_state.errors
            final_output = result_state.final_output
            analysis_report = result_state.analysis_report
            match_report = result_state.match_report
            current_step = result_state.current_step
        
        # Extract results
        return {
            "success": len(errors) == 0 and analysis_report is not None,
            "final_output": final_output,
            "analysis_report": analysis_report,
            "match_report": match_report,
            "errors": errors,
            "current_step": current_step
        }