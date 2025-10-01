"""
Streamlit App for CV Adaptor
Simplified interface with input/output and progress tracking
"""
import streamlit as st
import tempfile
import os
from pathlib import Path

from config import Config, LLMConfig, LLMProvider
from workflow import CVAdaptationWorkflow
from utils import extract_text_from_file, scrape_job_description, format_adapted_cv


def load_config_from_ui() -> Config:
    """Load configuration from Streamlit sidebar"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider selection
        provider = st.selectbox(
            "LLM Provider",
            options=["ollama", "openai", "anthropic", "groq"],
            help="Select your LLM provider"
        )
        
        config = Config()
        config.llm.provider = LLMProvider(provider)
        
        if provider == "ollama":
            config.llm.ollama_base_url = st.text_input(
                "Ollama URL",
                value="http://localhost:11434",
                help="URL where Ollama is running"
            )
            config.llm.ollama_model = st.text_input(
                "Ollama Model",
                value="llama3.2:3b",
                help="e.g., llama3.2:3b, llama3.1:8b, qwen2.5-coder:7b"
            )
        
        elif provider == "openai":
            config.llm.openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Your OpenAI API key"
            )
            config.llm.openai_model = st.text_input(
                "OpenAI Model",
                value="gpt-4o-mini",
                help="e.g., gpt-4o-mini, gpt-4o"
            )
        
        elif provider == "anthropic":
            config.llm.anthropic_api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Your Anthropic API key"
            )
            config.llm.anthropic_model = st.text_input(
                "Anthropic Model",
                value="claude-3-5-sonnet-20241022",
                help="e.g., claude-3-5-sonnet-20241022"
            )
        
        elif provider == "groq":
            config.llm.groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Your Groq API key"
            )
            config.llm.groq_model = st.text_input(
                "Groq Model",
                value="llama-3.1-8b-instant",
                help="e.g., llama-3.1-8b-instant"
            )
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            config.llm.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            config.workflow.enable_qa_loop = st.checkbox(
                "Enable QA Self-Correction Loop",
                value=True,
                help="Allow QA agent to trigger rewriting if issues found"
            )
            
            config.workflow.max_qa_iterations = st.number_input(
                "Max QA Iterations",
                min_value=1,
                max_value=5,
                value=2,
                help="Maximum number of QA refinement loops"
            )
            
            config.workflow.min_relevance_score = st.number_input(
                "Min Relevance Score",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                help="Minimum score to proceed with CV adaptation"
            )
        
        return config


def main():
    st.set_page_config(
        page_title="CV Adaptor AI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 CV Adaptor AI")
    st.markdown("**Intelligent CV Adaptation using Multi-Agent LLM System**")
    st.markdown("---")
    
    # Load configuration
    config = load_config_from_ui()
    
    # Initialize session state
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = None
    if "job_description" not in st.session_state:
        st.session_state.job_description = None
    if "result" not in st.session_state:
        st.session_state.result = None
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    
    st.header("📥 Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Your CV")
        
        uploaded_file = st.file_uploader(
            "Choose CV file",
            type=["pdf", "docx", "txt"],
            help="Upload your CV in PDF, DOCX, or TXT format"
        )
        
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                cv_text = extract_text_from_file(tmp_path, uploaded_file.name)
                os.unlink(tmp_path)
                
                if cv_text:
                    st.session_state.cv_text = cv_text
                    st.success(f"✅ CV uploaded: {uploaded_file.name}")
                    
                    with st.expander("Preview CV Text"):
                        st.text_area("CV Content", cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text, height=200, disabled=True)
                else:
                    st.error("❌ Failed to extract text from CV")
            
            except Exception as e:
                st.error(f"❌ Error processing CV: {str(e)}")
    
    with col2:
        st.subheader("2. Provide Job Description")
        
        # Option 1: URL
        job_url = st.text_input(
            "Job Posting URL",
            placeholder="https://example.com/job-posting",
            help="Enter the URL of the job posting"
        )
        
        if job_url and st.button("🌐 Scrape Job Description"):
            with st.spinner("Scraping job description..."):
                try:
                    jd_text = scrape_job_description(job_url)
                    if jd_text:
                        st.session_state.job_description = jd_text
                        st.success("✅ Job description scraped successfully!")
                    else:
                        st.error("❌ Failed to scrape job description")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Option 2: Manual paste
        st.markdown("**OR**")
        
        manual_jd = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Paste the full job description here...",
            help="Alternatively, paste the job description directly"
        )
        
        if manual_jd:
            st.session_state.job_description = manual_jd
            st.success("✅ Job description ready!")
    
    st.markdown("---")
    
    # ========================================================================
    # PROCESS SECTION
    # ========================================================================
    
    if st.session_state.cv_text and st.session_state.job_description:
        st.header("🤖 Process")
        
        if st.button("✨ Adapt CV to Job", type="primary", use_container_width=True):
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize workflow
                    status_text.text("Initializing multi-agent workflow...")
                    progress_bar.progress(10)
                    
                    workflow = CVAdaptationWorkflow(config)
                    
                    # Define progress callback
                    def update_progress(message: str):
                        status_text.text(f"🔄 {message}")
                    
                    # Phase 1: Parsing
                    status_text.text("📝 Phase 1/4: Parsing documents...")
                    progress_bar.progress(20)
                    
                    # Phase 2: Scoring
                    status_text.text("📊 Phase 2/4: Analyzing match and scoring...")
                    progress_bar.progress(40)
                    
                    # Phase 3: Rewriting
                    status_text.text("✍️ Phase 3/4: Rewriting CV sections...")
                    progress_bar.progress(60)
                    
                    # Phase 4: QA
                    status_text.text("✅ Phase 4/4: Quality assurance validation...")
                    progress_bar.progress(80)
                    
                    # Run workflow
                    result = workflow.run(
                        st.session_state.cv_text,
                        st.session_state.job_description,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ CV adaptation completed!")
                    
                    # Store result
                    st.session_state.result = result
                    
                    if result["success"]:
                        st.success("🎉 CV successfully adapted to job requirements!")
                    else:
                        st.warning(f"⚠️ Adaptation completed with issues: {', '.join(result['errors'])}")
                
                except Exception as e:
                    st.error(f"❌ Error during adaptation: {str(e)}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())
        
        st.markdown("---")
    
    # ========================================================================
    # OUTPUT SECTION
    # ========================================================================
    
    if st.session_state.result:
        st.header("📤 Output")
        
        result = st.session_state.result
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if result.get("match_report"):
                score = result["match_report"].relevance_score
                st.metric("Relevance Score", f"{score:.1f}%")
            else:
                st.metric("Relevance Score", "N/A")
        
        with col2:
            if result.get("qa_report"):
                qa_score = result["qa_report"].overall_score
                st.metric("QA Score", f"{qa_score:.1f}%")
            else:
                st.metric("QA Score", "N/A")
        
        with col3:
            st.metric("QA Iterations", result.get("qa_iterations", 0))
        
        with col4:
            qa_passed = result.get("qa_report", {}).passed if result.get("qa_report") else False
            st.metric("QA Status", "✅ Passed" if qa_passed else "⚠️ Issues")
        
        st.markdown("---")
        
        # Adapted CV
        if result.get("adapted_cv"):
            st.subheader("📄 Adapted CV")
            
            adapted_cv_text = format_adapted_cv(result["adapted_cv"])
            
            # Download button
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label="📥 Download CV",
                    data=adapted_cv_text,
                    file_name="adapted_cv.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Display adapted CV
            st.text_area("Adapted CV Content", adapted_cv_text, height=400)
        
        # Match Report
        if result.get("match_report"):
            with st.expander("📊 Match Analysis Report"):
                match_report = result["match_report"]
                
                st.write(f"**Recommendation:** {match_report.recommendation}")
                if match_report.reasoning:
                    st.write(f"**Reasoning:** {match_report.reasoning}")
                
                st.write("**Matched Skills:**")
                st.write(", ".join(match_report.matched_skills) if match_report.matched_skills else "None")
                
                st.write("**Skill Gaps:**")
                if match_report.skill_gaps:
                    for gap in match_report.skill_gaps:
                        st.write(f"- {gap.skill} ({gap.importance})")
                else:
                    st.write("No significant gaps")
                
                st.write("**Target Keywords:**")
                st.write(", ".join(match_report.target_keywords) if match_report.target_keywords else "None")
        
        # QA Report
        if result.get("qa_report"):
            with st.expander("✅ Quality Assurance Report"):
                qa_report = result["qa_report"]
                
                st.write(f"**Overall Score:** {qa_report.overall_score:.1f}%")
                st.write(f"**Passed:** {'✅ Yes' if qa_report.passed else '❌ No'}")
                st.write(f"**Factual Consistency:** {'✅' if qa_report.factual_consistency_check else '❌'}")
                st.write(f"**Style Consistency:** {'✅' if qa_report.style_consistency_check else '❌'}")
                
                if qa_report.issues:
                    st.write("**Issues Found:**")
                    for issue in qa_report.issues:
                        severity_emoji = {"critical": "🔴", "major": "🟡", "minor": "🟢"}.get(issue.severity, "⚪")
                        st.write(f"{severity_emoji} **{issue.section}** ({issue.issue_type}): {issue.description}")
                        if issue.suggested_fix:
                            st.write(f"   *Suggested fix: {issue.suggested_fix}*")
                
                st.write("**Keywords Verified:**")
                st.write(", ".join(qa_report.keywords_verified) if qa_report.keywords_verified else "None")
                
                if qa_report.missing_keywords:
                    st.write("**Missing Keywords:**")
                    st.write(", ".join(qa_report.missing_keywords))
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>CV Adaptor AI</strong> - Powered by Multi-Agent LLM System</p>
            <p>Uses LangGraph orchestration with Parser, Scoring, Rewriter, and QA agents</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()