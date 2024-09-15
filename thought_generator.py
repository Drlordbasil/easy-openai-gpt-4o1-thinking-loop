import time
from typing import List, Dict, Any, Tuple

class ThoughtGenerator:
    def __init__(self, structured_response_generator):
        self.structured_response_generator = structured_response_generator

    def generate_thoughts(self, initial_prompt: str, research_summary: Dict[str, Any], max_thoughts: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        thoughts = []
        start_time = time.time()
        
        thought_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The main content of the thought"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the thought"},
                "continue_thinking": {"type": "boolean", "description": "Whether to continue the thought process"}
            },
            "required": ["content", "key_points", "continue_thinking"]
        }

        for i in range(max_thoughts):
            thought_prompt = self._create_thought_prompt(initial_prompt, research_summary, thoughts)
            
            messages = [
                {"role": "system", "content": "You are an AI assistant generating thoughts on a given topic. Provide insightful and relevant thoughts."},
                {"role": "user", "content": thought_prompt}
            ]

            try:
                thought = self.structured_response_generator.generate(messages, thought_schema)
                thoughts.append(thought)
                
                if not thought['continue_thinking']:
                    break
            except Exception as e:
                print(f"Error generating thought {i + 1}: {e}")
                break

        thinking_time = time.time() - start_time
        return thoughts, thinking_time

    def _create_thought_prompt(self, initial_prompt: str, research_summary: Dict[str, Any], previous_thoughts: List[Dict[str, Any]]) -> str:
        prompt = f"Initial prompt: {initial_prompt}\n\nResearch summary: {research_summary['summary']}\n\n"
        
        if previous_thoughts:
            prompt += "Previous thoughts:\n"
            for i, thought in enumerate(previous_thoughts, 1):
                prompt += f"Thought {i}: {thought['content']}\n"
        
        prompt += "\nGenerate the next thought in this sequence. If you believe the thought process is complete, set 'continue_thinking' to false."
        return prompt

    def generate_response(self, thoughts: List[Dict[str, Any]], initial_prompt: str) -> Dict[str, Any]:
        response_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A comprehensive response based on the thoughts"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the response"}
            },
            "required": ["content", "key_points"]
        }

        response_prompt = self._create_response_prompt(thoughts, initial_prompt)
        
        messages = [
            {"role": "system", "content": "You are an AI assistant generating a comprehensive response based on a series of thoughts. Synthesize the information and provide a coherent answer."},
            {"role": "user", "content": response_prompt}
        ]

        try:
            response = self.structured_response_generator.generate(messages, response_schema)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"content": "Error generating response", "key_points": []}

    def _create_response_prompt(self, thoughts: List[Dict[str, Any]], initial_prompt: str) -> str:
        prompt = f"Initial prompt: {initial_prompt}\n\nThoughts:\n"
        for i, thought in enumerate(thoughts, 1):
            prompt += f"Thought {i}: {thought['content']}\n"
        prompt += "\nBased on these thoughts, generate a comprehensive response to the initial prompt."
        return prompt

    def reflect(self, thoughts: List[Dict[str, Any]], response: Dict[str, Any], thinking_time: float) -> Dict[str, Any]:
        reflection_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A reflection on the thought process and response"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the reflection"},
                "areas_for_improvement": {"type": "array", "items": {"type": "string"}, "description": "Areas where the thought process or response could be improved"},
                "confidence_level": {"type": "number", "description": "Confidence level in the overall process and response (0-1)"}
            },
            "required": ["content", "key_points", "areas_for_improvement", "confidence_level"]
        }

        reflection_prompt = self._create_reflection_prompt(thoughts, response, thinking_time)
        
        messages = [
            {"role": "system", "content": "You are an AI assistant reflecting on a thought process and response. Provide insights on the process and suggest improvements."},
            {"role": "user", "content": reflection_prompt}
        ]

        try:
            reflection = self.structured_response_generator.generate(messages, reflection_schema)
            return reflection
        except Exception as e:
            print(f"Error generating reflection: {e}")
            return {"content": "Error generating reflection", "key_points": [], "areas_for_improvement": [], "confidence_level": 0}

    def _create_reflection_prompt(self, thoughts: List[Dict[str, Any]], response: Dict[str, Any], thinking_time: float) -> str:
        prompt = f"Thought process:\n"
        for i, thought in enumerate(thoughts, 1):
            prompt += f"Thought {i}: {thought['content']}\n"
        prompt += f"\nResponse: {response['content']}\n"
        prompt += f"\nThinking time: {thinking_time:.2f} seconds\n"
        prompt += "\nReflect on the thought process and response. Identify strengths, weaknesses, and areas for improvement. Assign a confidence level to the overall process and response."
        return prompt