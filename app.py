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
from utils import extract_text_from_file, scrape_job_description


def format_analysis_report(analysis, match_report=None):
    """Format CV analysis report as plain text for download"""
    lines = []
    lines.append("=" * 80)
    lines.append("CV ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall Assessment
    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 80)
    lines.append(analysis.overall_assessment)
    lines.append("")
    
    if match_report:
        lines.append(f"Relevance Score: {match_report.relevance_score:.1f}%")
        lines.append("")
    
    # Quick Wins
    if analysis.quick_wins:
        lines.append("QUICK WINS")
        lines.append("-" * 80)
        for i, win in enumerate(analysis.quick_wins, 1):
            lines.append(f"{i}. {win}")
        lines.append("")
    
    # Critical Gaps
    if analysis.critical_gaps:
        lines.append("CRITICAL GAPS")
        lines.append("-" * 80)
        for gap in analysis.critical_gaps:
            lines.append(f"• {gap}")
        lines.append("")
    
    # Strengths
    if analysis.strengths_to_emphasize:
        lines.append("STRENGTHS TO EMPHASIZE")
        lines.append("-" * 80)
        for strength in analysis.strengths_to_emphasize:
            lines.append(f"• {strength}")
        lines.append("")
    
    # Section Analyses
    lines.append("DETAILED SECTION ANALYSIS")
    lines.append("=" * 80)
    
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_sections = sorted(
        analysis.section_analyses,
        key=lambda x: priority_order.get(x.priority.lower(), 3)
    )
    
    for section in sorted_sections:
        lines.append("")
        lines.append(f"{section.section_name.upper()} [Priority: {section.priority.upper()}]")
        lines.append("-" * 80)
        lines.append(f"Current Status: {section.current_status}")
        lines.append("")
        
        if section.required_changes:
            lines.append("Required Changes:")
            for change in section.required_changes:
                lines.append(f"  • {change}")
            lines.append("")
        
        if section.suggested_additions:
            lines.append("Suggested Additions:")
            for addition in section.suggested_additions:
                lines.append(f"  • {addition}")
            lines.append("")
        
        if section.keywords_to_add:
            lines.append(f"Keywords to Add: {', '.join(section.keywords_to_add)}")
            lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


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
            
            config.workflow.min_relevance_score = st.number_input(
                "Min Relevance Score",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                help="Minimum score to proceed with CV analysis"
            )
        
        return config


def main():
    st.set_page_config(
        page_title="CV Adaptor AI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 CV Analysis AI")
    st.markdown("**Intelligent CV Analysis using Multi-Agent LLM System**")
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
        
        if st.button("📊 Analyze CV for Job", type="primary", use_container_width=True):
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
                    status_text.text("📝 Phase 1/3: Parsing documents...")
                    progress_bar.progress(20)
                    
                    # Phase 2: Scoring
                    status_text.text("📊 Phase 2/3: Analyzing match and scoring...")
                    progress_bar.progress(50)
                    
                    # Phase 3: Analysis
                    status_text.text("🔍 Phase 3/3: Generating analysis report...")
                    progress_bar.progress(80)
                    
                    # Run workflow
                    result = workflow.run(
                        st.session_state.cv_text,
                        st.session_state.job_description,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ CV analysis completed!")
                    
                    # Store result
                    st.session_state.result = result
                    
                    if result["success"]:
                        st.success("🎉 CV analysis generated successfully!")
                    else:
                        st.warning(f"⚠️ Analysis completed with issues: {', '.join(result['errors'])}")
                
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result.get("match_report"):
                score = result["match_report"].relevance_score
                st.metric("Relevance Score", f"{score:.1f}%")
            else:
                st.metric("Relevance Score", "N/A")
        
        with col2:
            if result.get("match_report"):
                matched_count = len(result["match_report"].matched_skills)
                st.metric("Matched Skills", matched_count)
            else:
                st.metric("Matched Skills", "N/A")
        
        with col3:
            if result.get("match_report"):
                gap_count = len(result["match_report"].skill_gaps)
                st.metric("Skill Gaps", gap_count)
            else:
                st.metric("Skill Gaps", "N/A")
        
        st.markdown("---")
        
        # CV Analysis Report
        if result.get("analysis_report"):
            st.subheader("📋 CV Analysis Report")
            
            analysis = result["analysis_report"]
            
            # Overall Assessment
            st.markdown("### 🎯 Overall Assessment")
            st.info(analysis.overall_assessment)
            
            # Quick Wins
            if analysis.quick_wins:
                st.markdown("### ⚡ Quick Wins")
                st.markdown("*Easy improvements you can make right away:*")
                for i, win in enumerate(analysis.quick_wins, 1):
                    st.markdown(f"{i}. {win}")
            
            # Critical Gaps
            if analysis.critical_gaps:
                st.markdown("### 🚨 Critical Gaps")
                st.markdown("*Must-have requirements that are missing:*")
                for gap in analysis.critical_gaps:
                    st.error(f"❌ {gap}")
            
            # Strengths to Emphasize
            if analysis.strengths_to_emphasize:
                st.markdown("### ✅ Strengths to Emphasize")
                st.markdown("*What's already good in your CV:*")
                for strength in analysis.strengths_to_emphasize:
                    st.success(f"✓ {strength}")
            
            st.markdown("---")
            
            # Section-by-Section Analysis
            st.markdown("### 📊 Detailed Section Analysis")
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            sorted_sections = sorted(
                analysis.section_analyses,
                key=lambda x: priority_order.get(x.priority.lower(), 3)
            )
            
            for section in sorted_sections:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(section.priority.lower(), "⚪")
                
                with st.expander(f"{priority_emoji} {section.section_name} (Priority: {section.priority.upper()})", expanded=(section.priority.lower() == "high")):
                    st.markdown(f"**Current Status:** {section.current_status}")
                    
                    if section.required_changes:
                        st.markdown("**Required Changes:**")
                        for change in section.required_changes:
                            st.markdown(f"- {change}")
                    
                    if section.suggested_additions:
                        st.markdown("**Suggested Additions:**")
                        for addition in section.suggested_additions:
                            st.markdown(f"- {addition}")
                    
                    if section.keywords_to_add:
                        st.markdown("**Keywords to Add:**")
                        st.code(", ".join(section.keywords_to_add))
            
            # Download Analysis Report
            st.markdown("---")
            report_text = format_analysis_report(analysis, result.get("match_report"))
            st.download_button(
                label="📥 Download Analysis Report",
                data=report_text,
                file_name="cv_analysis_report.txt",
                mime="text/plain",
                use_container_width=False
            )
        
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>CV Analysis AI</strong> - Powered by Multi-Agent LLM System</p>
            <p>Uses LangGraph orchestration with Parser, Scoring, and Analysis agents</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()