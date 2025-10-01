"""
Simple example script to test CV Adaptor
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
    print("🚀 CV Adaptor Example")
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
    config.workflow.enable_qa_loop = True
    config.workflow.max_qa_iterations = 2
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
    
    # Run adaptation
    print("\n🤖 Running CV adaptation workflow...")
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
            print("\n✅ CV adaptation completed successfully!")
            
            # Display metrics
            print("\n📊 Results:")
            print(f"  • Relevance Score: {result['match_report'].relevance_score:.1f}%")
            
            if result.get("qa_report"):
                print(f"  • QA Score: {result['qa_report'].overall_score:.1f}%")
                print(f"  • QA Passed: {'✅ Yes' if result['qa_report'].passed else '❌ No'}")
            
            print(f"  • QA Iterations: {result['qa_iterations']}")
            
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
            
            # Display adapted CV summary
            if result.get("adapted_cv"):
                print("\n📄 Adapted CV Summary:")
                adapted_cv = result["adapted_cv"]
                print(f"  • Name: {adapted_cv.contact.name}")
                print(f"  • Experience Entries: {len(adapted_cv.experience)}")
                print(f"  • Skills: {len(adapted_cv.skills)}")
                
                print("\n  New Professional Summary:")
                print(f"  {adapted_cv.summary[:200]}..." if len(adapted_cv.summary) > 200 else f"  {adapted_cv.summary}")
                
                # Save to file
                from utils import format_adapted_cv
                adapted_text = format_adapted_cv(adapted_cv)
                
                output_file = "adapted_cv_example.txt"
                with open(output_file, "w") as f:
                    f.write(adapted_text)
                
                print(f"\n💾 Full adapted CV saved to: {output_file}")
        
        else:
            print("\n❌ CV adaptation failed!")
            print(f"Errors: {', '.join(result['errors'])}")
    
    except Exception as e:
        print(f"\n❌ Error during adaptation: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
    
    print("\n" + "=" * 80)
    print("Done! Check the output above for results.")


if __name__ == "__main__":
    main()
