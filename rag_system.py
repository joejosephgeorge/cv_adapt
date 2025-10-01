"""
RAG (Retrieval-Augmented Generation) System for CV Adaptor
Implements vector database indexing and retrieval for grounding LLM generation
"""
from typing import List, Dict, Any, Optional
from config import RAGConfig
import chromadb
from chromadb.config import Settings
import hashlib


class RAGSystem:
    """RAG system using ChromaDB for vector storage and retrieval"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(
            path=config.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Separate collections for CV and JD
        self.cv_collection = self.client.get_or_create_collection(
            name="cv_facts",
            metadata={"description": "CV facts for RAG grounding"}
        )
        self.jd_collection = self.client.get_or_create_collection(
            name="job_requirements",
            metadata={"description": "Job requirements for RAG grounding"}
        )
    
    def index_cv_facts(self, candidate_profile: Dict[str, Any]) -> None:
        """
        Index CV facts into vector database
        
        Args:
            candidate_profile: Structured candidate profile (CandidateProfileSchema)
        """
        documents = []
        metadatas = []
        ids = []
        
        # Index summary
        if candidate_profile.get("summary"):
            doc = f"Professional Summary: {candidate_profile['summary']}"
            documents.append(doc)
            metadatas.append({"type": "summary"})
            ids.append(self._generate_id(doc))
        
        # Index experience with achievements and metrics
        for idx, exp in enumerate(candidate_profile.get("experience", [])):
            # Main experience description
            doc = f"Position: {exp.get('position')} at {exp.get('company')} ({exp.get('duration')}). {exp.get('description', '')}"
            documents.append(doc)
            metadatas.append({"type": "experience", "position": exp.get("position")})
            ids.append(self._generate_id(doc))
            
            # Index achievements separately for better retrieval
            for achievement in exp.get("achievements", []):
                doc = f"Achievement at {exp.get('company')}: {achievement}"
                documents.append(doc)
                metadatas.append({"type": "achievement", "company": exp.get("company")})
                ids.append(self._generate_id(doc))
            
            # Index metrics
            for metric in exp.get("metrics", []):
                doc = f"Metric from {exp.get('company')}: {metric}"
                documents.append(doc)
                metadatas.append({"type": "metric", "company": exp.get("company")})
                ids.append(self._generate_id(doc))
            
            # Index skills used
            if exp.get("skills_used"):
                doc = f"Skills used at {exp.get('company')}: {', '.join(exp.get('skills_used', []))}"
                documents.append(doc)
                metadatas.append({"type": "skills", "company": exp.get("company")})
                ids.append(self._generate_id(doc))
        
        # Index education
        for edu in candidate_profile.get("education", []):
            doc = f"Education: {edu.get('degree')} in {edu.get('field', 'N/A')} from {edu.get('institution')} ({edu.get('duration')})"
            documents.append(doc)
            metadatas.append({"type": "education"})
            ids.append(self._generate_id(doc))
        
        # Index skills
        if candidate_profile.get("skills"):
            doc = f"Technical Skills: {', '.join(candidate_profile.get('skills', []))}"
            documents.append(doc)
            metadatas.append({"type": "skills"})
            ids.append(self._generate_id(doc))
        
        # Index certifications
        for cert in candidate_profile.get("certifications", []):
            doc = f"Certification: {cert}"
            documents.append(doc)
            metadatas.append({"type": "certification"})
            ids.append(self._generate_id(doc))
        
        # Index projects
        for project in candidate_profile.get("projects", []):
            doc = f"Project: {project}"
            documents.append(doc)
            metadatas.append({"type": "project"})
            ids.append(self._generate_id(doc))
        
        # Clear existing CV data and add new
        try:
            self.cv_collection.delete(where={})
        except:
            pass
        
        if documents:
            self.cv_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    def index_job_requirements(self, job_requirements: Dict[str, Any]) -> None:
        """
        Index job requirements into vector database
        
        Args:
            job_requirements: Structured job requirements (JobRequirementSchema)
        """
        documents = []
        metadatas = []
        ids = []
        
        # Index title and basic info
        doc = f"Job Title: {job_requirements.get('title')} at {job_requirements.get('company', 'N/A')}. Experience Level: {job_requirements.get('experience_level')}"
        documents.append(doc)
        metadatas.append({"type": "basic_info"})
        ids.append(self._generate_id(doc))
        
        # Index requirements
        for req in job_requirements.get("requirements", []):
            doc = f"{req.get('category')} ({req.get('importance')}): {req.get('requirement')}"
            if req.get("keywords"):
                doc += f" Keywords: {', '.join(req.get('keywords', []))}"
            documents.append(doc)
            metadatas.append({"type": "requirement", "category": req.get("category")})
            ids.append(self._generate_id(doc))
        
        # Index key skills
        if job_requirements.get("key_skills"):
            doc = f"Required Skills: {', '.join(job_requirements.get('key_skills', []))}"
            documents.append(doc)
            metadatas.append({"type": "skills"})
            ids.append(self._generate_id(doc))
        
        # Index responsibilities
        for responsibility in job_requirements.get("responsibilities", []):
            doc = f"Responsibility: {responsibility}"
            documents.append(doc)
            metadatas.append({"type": "responsibility"})
            ids.append(self._generate_id(doc))
        
        # Clear existing JD data and add new
        try:
            self.jd_collection.delete(where={})
        except:
            pass
        
        if documents:
            self.jd_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    def retrieve_cv_facts(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Retrieve relevant CV facts based on query
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant CV facts
        """
        k = top_k or self.config.top_k
        
        try:
            results = self.cv_collection.query(
                query_texts=[query],
                n_results=min(k, self.cv_collection.count())
            )
            
            if results and results["documents"]:
                return results["documents"][0]
        except:
            pass
        
        return []
    
    def retrieve_job_requirements(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Retrieve relevant job requirements based on query
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant job requirements
        """
        k = top_k or self.config.top_k
        
        try:
            results = self.jd_collection.query(
                query_texts=[query],
                n_results=min(k, self.jd_collection.count())
            )
            
            if results and results["documents"]:
                return results["documents"][0]
        except:
            pass
        
        return []
    
    def retrieve_multi_query(
        self, 
        queries: List[str], 
        collection: str = "cv"
    ) -> List[str]:
        """
        RAG Fusion: Retrieve using multiple queries and merge results
        
        Args:
            queries: List of related queries
            collection: "cv" or "jd"
        
        Returns:
            Merged list of relevant documents
        """
        all_results = []
        seen = set()
        
        for query in queries:
            if collection == "cv":
                results = self.retrieve_cv_facts(query)
            else:
                results = self.retrieve_job_requirements(query)
            
            for result in results:
                if result not in seen:
                    all_results.append(result)
                    seen.add(result)
        
        return all_results
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for document"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_all(self) -> None:
        """Clear all indexed data"""
        try:
            self.cv_collection.delete(where={})
            self.jd_collection.delete(where={})
        except:
            pass
