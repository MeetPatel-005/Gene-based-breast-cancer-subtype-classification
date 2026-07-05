import requests
import json
from src.agent_prompts import (
    ANALYZE_PROMPT, SYNTHESIZE_PROMPT, ROUTE_PROMPT,
    CHAT_SYSTEM_PROMPT, CHAT_CONTEXT_BLOCK,
)

class ClinicalAgent:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.url = ollama_url
        self.model = "mistral"
    
    def _query_llm(self, prompt: str) -> str:
        """Query local Ollama instance."""
        response = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=30
        )
        return response.json()["response"]
    
    def analyze(self, subtype: str, confidence: float, probs: dict, biomarkers: list) -> dict:
        """Step 1: Analyze prediction vs biomarkers."""
        prompt = ANALYZE_PROMPT.format(
            subtype=subtype, 
            confidence=f"{confidence:.2%}",
            probs=probs,
            biomarkers=", ".join(biomarkers) or "None detected"
        )
        response = self._query_llm(prompt)
        try:
            return json.loads(response)
        except:
            return {"alignment": "medium", "explanation": response}
    
    def synthesize(self, analysis: dict, subtype: str) -> dict:
        """Step 2: Generate clinical summary."""
        prompt = SYNTHESIZE_PROMPT.format(
            analysis=json.dumps(analysis),
            subtype=subtype
        )
        response = self._query_llm(prompt)
        try:
            return json.loads(response)
        except:
            return {"risk_level": "intermediate", "prognosis": response, "key_insights": ""}
    
    def route(self, subtype: str, risk_level: str) -> dict:
        """Step 3: Route to therapy."""
        prompt = ROUTE_PROMPT.format(subtype=subtype, risk_level=risk_level)
        response = self._query_llm(prompt)
        try:
            return json.loads(response)
        except:
            return {"primary_therapy": response, "alternatives": [], "rationale": ""}
    
    def run_full_agent(self, model_output: dict) -> dict:
        """Execute 3-step agentic loop."""
        subtype = model_output["subtype"]
        confidence = model_output["confidence"]
        probs = model_output["probabilities"]
        biomarkers = model_output.get("detected_biomarkers", [])
        
        # Step 1
        analysis = self.analyze(subtype, confidence, probs, biomarkers)
        
        # Step 2
        summary = self.synthesize(analysis, subtype)
        
        # Step 3
        therapy = self.route(subtype, summary["risk_level"])
        
        return {
            "subtype": subtype,
            "confidence": confidence,
            "analysis": analysis,
            "clinical_summary": summary,
            "therapy_routes": therapy
        }

    def chat(self, message: str, history: list, context: dict = None) -> str:
        """Conversational chat with optional patient context awareness."""
        context_block = ""
        if context:
            agent = context.get("agent_analysis", {})
            summary = agent.get("clinical_summary", {})
            therapy = agent.get("therapy_routes", {})
            analysis = agent.get("analysis", {})
            context_block = CHAT_CONTEXT_BLOCK.format(
                subtype=context.get("subtype", "Unknown"),
                confidence=f"{context.get('confidence', 0):.1%}",
                risk_level=summary.get("risk_level", "unknown"),
                primary_therapy=therapy.get("primary_therapy", "not determined"),
                biomarkers=", ".join(context.get("detected_biomarkers", [])) or "None",
                alignment_explanation=analysis.get("explanation", "Not available"),
            )

        system = CHAT_SYSTEM_PROMPT.format(context_block=context_block)

        # Include last 6 turns to stay within token limits
        history_text = ""
        for turn in history[-6:]:
            history_text += f"Human: {turn['user']}\nAssistant: {turn['bot']}\n\n"

        full_prompt = f"{system}\n\n{history_text}Human: {message}\nAssistant:"
        return self._query_llm(full_prompt)