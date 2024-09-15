import time
import random
from typing import List, Dict, Any, Tuple

class ThoughtGenerator:
    def __init__(self, structured_response_generator, web_researcher):
        self.structured_response_generator = structured_response_generator
        self.web_researcher = web_researcher

    def generate_thoughts(self, initial_prompt: str, initial_research_summary: Dict[str, Any], max_thoughts: int = 20, max_thinking_time: float = 300) -> Tuple[List[Dict[str, Any]], float]:
        thoughts = []
        start_time = time.time()
        
        thought_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The main content of the thought"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the thought"},
                "reasoning": {"type": "string", "description": "Detailed reasoning behind the thought"},
                "continue_thinking": {"type": "boolean", "description": "Whether to continue the thought process"},
                "conduct_research": {"type": "boolean", "description": "Whether additional research is needed"}
            },
            "required": ["content", "key_points", "reasoning", "continue_thinking", "conduct_research"]
        }

        current_research_summary = initial_research_summary

        while len(thoughts) < max_thoughts and (time.time() - start_time) < max_thinking_time:
            thought_prompt = self._create_thought_prompt(initial_prompt, current_research_summary, thoughts)
            
            messages = [
                {"role": "system", "content": "You are an AI assistant generating deep, insightful thoughts on a given topic. Provide thorough reasoning and decide if more thinking or research is needed."},
                {"role": "user", "content": thought_prompt}
            ]

            try:
                thought = self.structured_response_generator.generate(messages, thought_schema)
                thoughts.append(thought)
                
                if thought['conduct_research']:
                    research_query = self._generate_research_query(initial_prompt, thoughts)
                    new_research = self.web_researcher.conduct_research(research_query)
                    current_research_summary = self._merge_research(current_research_summary, new_research)
                
                if not thought['continue_thinking']:
                    break
            except Exception as e:
                print(f"Error generating thought {len(thoughts) + 1}: {e}")
                break

        thinking_time = time.time() - start_time
        return thoughts, thinking_time

    def _create_thought_prompt(self, initial_prompt: str, research_summary: Dict[str, Any], previous_thoughts: List[Dict[str, Any]]) -> str:
        prompt = f"Initial prompt: {initial_prompt}\n\nCurrent research summary: {research_summary['summary']}\n\n"
        
        if previous_thoughts:
            prompt += "Previous thoughts:\n"
            for i, thought in enumerate(previous_thoughts, 1):
                prompt += f"Thought {i}:\nContent: {thought['content']}\nReasoning: {thought['reasoning']}\n\n"
        
        prompt += "\nGenerate the next thought in this sequence. Provide detailed reasoning for your thought. Decide if more thinking is needed (continue_thinking) and if additional research would be beneficial (conduct_research)."
        return prompt

    def _generate_research_query(self, initial_prompt: str, thoughts: List[Dict[str, Any]]) -> str:
        query_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused research query based on the current thoughts and initial prompt"}
            },
            "required": ["query"]
        }

        query_prompt = f"Based on the initial prompt: '{initial_prompt}' and the following thoughts:\n"
        for i, thought in enumerate(thoughts[-3:], 1):  # Consider only the last 3 thoughts for brevity
            query_prompt += f"Thought {i}: {thought['content']}\n"
        query_prompt += "\nGenerate a focused research query to gather more information on the most pressing aspect of the current thought process."

        messages = [
            {"role": "system", "content": "You are an AI assistant generating a focused research query based on the current thought process."},
            {"role": "user", "content": query_prompt}
        ]

        try:
            query_response = self.structured_response_generator.generate(messages, query_schema)
            return query_response['query']
        except Exception as e:
            print(f"Error generating research query: {e}")
            return initial_prompt  # Fallback to the initial prompt if query generation fails

    def _merge_research(self, original_research: Dict[str, Any], new_research: Dict[str, Any]) -> Dict[str, Any]:
        merged_summary = f"{original_research['summary']}\n\nAdditional research:\n{new_research['summary']}"
        merged_key_points = list(set(original_research['key_points'] + new_research['key_points']))
        merged_sources = list(set(original_research.get('sources', []) + new_research.get('sources', [])))

        return {
            "summary": merged_summary,
            "key_points": merged_key_points,
            "sources": merged_sources
        }

    def generate_response(self, thoughts: List[Dict[str, Any]], initial_prompt: str) -> Dict[str, Any]:
        response_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A comprehensive response based on the thoughts"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the response"},
                "reasoning": {"type": "string", "description": "Detailed reasoning behind the response"}
            },
            "required": ["content", "key_points", "reasoning"]
        }

        response_prompt = self._create_response_prompt(thoughts, initial_prompt)
        
        messages = [
            {"role": "system", "content": "You are an AI assistant generating a comprehensive response based on a series of thoughts. Synthesize the information, provide a coherent answer, and include detailed reasoning."},
            {"role": "user", "content": response_prompt}
        ]

        try:
            response = self.structured_response_generator.generate(messages, response_schema)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"content": "Error generating response", "key_points": [], "reasoning": ""}

    def _create_response_prompt(self, thoughts: List[Dict[str, Any]], initial_prompt: str) -> str:
        prompt = f"Initial prompt: {initial_prompt}\n\nThoughts:\n"
        for i, thought in enumerate(thoughts, 1):
            prompt += f"Thought {i}:\nContent: {thought['content']}\nReasoning: {thought['reasoning']}\n\n"
        prompt += "\nBased on these thoughts, generate a comprehensive response to the initial prompt. Include detailed reasoning for your response."
        return prompt

    def reflect(self, thoughts: List[Dict[str, Any]], response: Dict[str, Any], thinking_time: float) -> Dict[str, Any]:
        reflection_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A reflection on the thought process and response"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the reflection"},
                "areas_for_improvement": {"type": "array", "items": {"type": "string"}, "description": "Areas where the thought process or response could be improved"},
                "confidence_level": {"type": "number", "description": "Confidence level in the overall process and response (0-1)"},
                "meta_cognition": {"type": "string", "description": "Analysis of the thinking process itself"}
            },
            "required": ["content", "key_points", "areas_for_improvement", "confidence_level", "meta_cognition"]
        }

        reflection_prompt = self._create_reflection_prompt(thoughts, response, thinking_time)
        
        messages = [
            {"role": "system", "content": "You are an AI assistant reflecting on a thought process and response. Provide deep insights on the process, suggest improvements, and analyze the thinking process itself."},
            {"role": "user", "content": reflection_prompt}
        ]

        try:
            reflection = self.structured_response_generator.generate(messages, reflection_schema)
            return reflection
        except Exception as e:
            print(f"Error generating reflection: {e}")
            return {"content": "Error generating reflection", "key_points": [], "areas_for_improvement": [], "confidence_level": 0, "meta_cognition": ""}

    def _create_reflection_prompt(self, thoughts: List[Dict[str, Any]], response: Dict[str, Any], thinking_time: float) -> str:
        prompt = f"Thought process:\n"
        for i, thought in enumerate(thoughts, 1):
            prompt += f"Thought {i}:\nContent: {thought['content']}\nReasoning: {thought['reasoning']}\n\n"
        prompt += f"\nFinal Response:\nContent: {response['content']}\nReasoning: {response['reasoning']}\n"
        prompt += f"\nThinking time: {thinking_time:.2f} seconds\n"
        prompt += "\nReflect on the thought process and response. Identify strengths, weaknesses, and areas for improvement. Analyze the thinking process itself (meta-cognition). Assign a confidence level to the overall process and response."
        return prompt