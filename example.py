"""
Simple example script to test CV Analysis
Run with: python example.py
"""
from config import Config, LLMProvider
from workflow import CVAdaptationWorkflow

# Sample CV text
SAMPLE_CV = """
John Doe
john.doe@email.com | (555) 123-4567 | San Francisco, CA
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

PROFESSIONAL SUMMARY
Software engineer with 5 years of experience in web development and cloud infrastructure.
Proficient in Python, JavaScript, and AWS. Strong problem-solving skills.

WORK EXPERIENCE
Senior Software Engineer | Tech Corp | 2021 - Present
- Developed microservices architecture serving 10M+ users
- Reduced deployment time by 60% through CI/CD automation
- Led team of 3 junior developers
- Technologies: Python, Docker, Kubernetes, AWS

Software Engineer | StartupXYZ | 2019 - 2021
- Built RESTful APIs for mobile and web applications
- Implemented authentication system with OAuth2
- Optimized database queries improving performance by 40%
- Technologies: Node.js, PostgreSQL, React

EDUCATION
Bachelor of Science in Computer Science
State University | 2015 - 2019 | GPA: 3.8

TECHNICAL SKILLS
Python, JavaScript, TypeScript, React, Node.js, Docker, Kubernetes, AWS, PostgreSQL, MongoDB, Git

CERTIFICATIONS
AWS Certified Solutions Architect - Associate
"""

# Sample Job Description
SAMPLE_JD = """
Senior Backend Engineer
TechInnovate Inc. | Remote | $150k - $200k

About the Role:
We're looking for a Senior Backend Engineer to join our platform team. You'll be building
scalable microservices and APIs that power our SaaS platform serving millions of users.

Required Skills:
- 5+ years of backend development experience
- Strong proficiency in Python and Django
- Experience with microservices architecture
- Docker and Kubernetes expertise
- AWS cloud services (EC2, S3, RDS, Lambda)
- PostgreSQL or MySQL databases
- RESTful API design
- CI/CD pipelines

Preferred Skills:
- GraphQL
- Redis caching
- Message queues (RabbitMQ, Kafka)
- Terraform infrastructure as code
- Experience with high-traffic systems (1M+ daily users)

Responsibilities:
- Design and implement scalable backend services
- Build and maintain APIs for web and mobile clients
- Optimize database performance and queries
- Implement monitoring and logging solutions
- Mentor junior developers
- Participate in code reviews and architecture discussions

Requirements:
- Bachelor's degree in Computer Science or related field
- Strong communication skills
- Experience with agile development
"""


def main():
    print("🚀 CV Analysis Example")
    print("=" * 80)
    
    # Initialize configuration
    print("\n📋 Initializing configuration...")
    config = Config()
    
    # Configure for Ollama (local)
    # Change this to use other providers
    config.llm.provider = LLMProvider.OLLAMA
    config.llm.ollama_base_url = "http://localhost:11434"
    config.llm.ollama_model = "llama3.2:3b"
    
    # Workflow settings
    config.workflow.min_relevance_score = 50.0
    
    print(f"✅ Using {config.llm.provider} with model {config.llm.ollama_model}")
    
    # Initialize workflow
    print("\n🔧 Initializing multi-agent workflow...")
    try:
        workflow = CVAdaptationWorkflow(config)
        print("✅ Workflow initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing workflow: {str(e)}")
        print("\n💡 Make sure Ollama is running: ollama serve")
        print("💡 And the model is pulled: ollama pull llama3.2:3b")
        return
    
    # Run analysis
    print("\n🤖 Running CV analysis workflow...")
    print("-" * 80)
    
    def progress_callback(message: str):
        print(f"  ⏳ {message}")
    
    try:
        result = workflow.run(
            SAMPLE_CV,
            SAMPLE_JD,
            progress_callback=progress_callback
        )
        
        print("-" * 80)
        
        if result["success"]:
            print("\n✅ CV analysis completed successfully!")
            
            # Display metrics
            print("\n📊 Results:")
            print(f"  • Relevance Score: {result['match_report'].relevance_score:.1f}%")
            
            # Display match analysis
            if result.get("match_report"):
                print("\n🎯 Match Analysis:")
                match_report = result["match_report"]
                print(f"  • Recommendation: {match_report.recommendation}")
                print(f"  • Matched Skills: {len(match_report.matched_skills)}")
                print(f"  • Skill Gaps: {len(match_report.skill_gaps)}")
                
                if match_report.skill_gaps:
                    print("\n  Missing Skills:")
                    for gap in match_report.skill_gaps[:5]:
                        print(f"    - {gap.skill} ({gap.importance})")
            
            # Display analysis report
            if result.get("analysis_report"):
                print("\n📋 CV Analysis Report:")
                analysis = result["analysis_report"]
                
                print(f"\n  Overall Assessment:")
                print(f"  {analysis.overall_assessment}")
                
                if analysis.quick_wins:
                    print(f"\n  ⚡ Quick Wins:")
                    for i, win in enumerate(analysis.quick_wins[:3], 1):
                        print(f"    {i}. {win}")
                
                if analysis.critical_gaps:
                    print(f"\n  🚨 Critical Gaps:")
                    for gap in analysis.critical_gaps[:3]:
                        print(f"    • {gap}")
                
                print(f"\n  📊 Section Analyses: {len(analysis.section_analyses)} sections")
                for section in analysis.section_analyses[:3]:
                    print(f"    • {section.section_name} [Priority: {section.priority}]")
                
                # Save to file
                output_file = "cv_analysis_report.txt"
                
                # Format report
                lines = []
                lines.append("=" * 80)
                lines.append("CV ANALYSIS REPORT")
                lines.append("=" * 80)
                lines.append("")
                lines.append(f"Overall Assessment: {analysis.overall_assessment}")
                lines.append("")
                
                if analysis.quick_wins:
                    lines.append("QUICK WINS:")
                    for win in analysis.quick_wins:
                        lines.append(f"  • {win}")
                    lines.append("")
                
                if analysis.critical_gaps:
                    lines.append("CRITICAL GAPS:")
                    for gap in analysis.critical_gaps:
                        lines.append(f"  • {gap}")
                    lines.append("")
                
                lines.append("SECTION ANALYSES:")
                for section in analysis.section_analyses:
                    lines.append(f"\n{section.section_name} [Priority: {section.priority}]")
                    lines.append(f"  Status: {section.current_status}")
                    if section.required_changes:
                        lines.append("  Changes:")
                        for change in section.required_changes:
                            lines.append(f"    - {change}")
                
                lines.append("\n" + "=" * 80)
                
                with open(output_file, "w") as f:
                    f.write("\n".join(lines))
                
                print(f"\n💾 Full analysis report saved to: {output_file}")
        
        else:
            print("\n❌ CV analysis failed!")
            print(f"Errors: {', '.join(result['errors'])}")
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
    
    print("\n" + "=" * 80)
    print("Done! Check the output above for results.")


if __name__ == "__main__":
    main()
