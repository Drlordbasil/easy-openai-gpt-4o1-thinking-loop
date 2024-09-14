import json
import time
from typing import List, Dict, Any

class ThoughtGenerator:
    def __init__(self, structured_response_generator):
        self.structured_response_generator = structured_response_generator

    def generate_thoughts(self, initial_prompt: str, research_summary: Dict[str, Any], max_iterations: int = 20, max_time: int = 300):
        thoughts = []
        messages = [
            {"role": "system", "content": """You are an advanced AI capable of deep, iterative thinking using a tree of thought approach. Generate comprehensive and structured thoughts on the given topic. 
            Your output should strictly follow the JSON schema provided. Engage in self-debate to determine when to stop thinking.
            
            Here's an example of a valid thought structure:
            {
                "content": "AGI could revolutionize healthcare through personalized medicine and accelerated drug discovery, potentially leading to longer lifespans and improved quality of life. However, it raises concerns about privacy, data security, and potential job displacement in the medical field.",
                "key_points": [
                    "Personalized medicine advancements",
                    "Accelerated drug discovery",
                    "Potential for longer lifespans",
                    "Improved quality of life",
                    "Privacy and data security concerns",
                    "Potential job displacement in healthcare"
                ],
                "sub_thoughts": [
                    {
                        "content": "Exploring the ethical implications of AGI in healthcare...",
                        "key_points": ["Ethical consideration 1", "Ethical consideration 2"],
                        "sub_thoughts": []
                    },
                    {
                        "content": "Analyzing the economic impact of AGI-driven healthcare innovations...",
                        "key_points": ["Economic impact 1", "Economic impact 2"],
                        "sub_thoughts": []
                    }
                ],
                "continue_thinking": true,
                "reasoning": "Further exploration is needed to fully understand the societal implications."
            }
            
            Ensure your response strictly adheres to this format, including nested sub_thoughts when necessary."""},
            {"role": "user", "content": f"Initial prompt: {initial_prompt}\n\nResearch summary: {research_summary['summary']}\n\nKey points from research: {', '.join(research_summary['key_points'])}"}
        ]

        thought_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A detailed exploration of the topic"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "A list of key points from the content"},
                "sub_thoughts": {
                    "type": "array",
                    "items": {
                        "$ref": "#"
                    },
                    "description": "Nested thoughts exploring sub-topics"
                },
                "continue_thinking": {"type": "boolean", "description": "Whether to continue the thought process"},
                "reasoning": {"type": "string", "description": "Reasoning for continuing or stopping the thought process"}
            },
            "required": ["content", "key_points", "sub_thoughts", "continue_thinking", "reasoning"]
        }

        start_time = time.time()
        for i in range(max_iterations):
            thought = self.structured_response_generator.generate(messages, thought_schema)
            thoughts.append(thought)
            
            if not thought['continue_thinking']:
                break
            
            messages.append({"role": "assistant", "content": json.dumps(thought)})
            messages.append({"role": "user", "content": f"""Thought {i+1} complete. Engage in self-debate to determine if further exploration is necessary. Consider:
            1. Have all key aspects of the topic been thoroughly explored?
            2. Are there any contradictions or gaps in the current thoughts?
            3. Could additional thinking lead to novel insights?
            4. Is the current depth of analysis sufficient for the complexity of the topic?

            Based on this self-debate, decide whether to continue thinking or conclude the process. Provide clear reasoning for your decision."""})
            
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                print(f"Maximum thinking time of {max_time} seconds reached. Concluding thought process.")
                break

        thinking_time = time.time() - start_time
        return thoughts, thinking_time

    def generate_response(self, thoughts: List[Dict[str, Any]], initial_prompt: str) -> Dict[str, Any]:
        response_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A coherent response to the initial prompt"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "A list of key points from the response"},
                "thought_process": {"type": "string", "description": "A brief explanation of the thought process that led to this response"}
            },
            "required": ["content", "key_points", "thought_process"]
        }

        response_prompt = f"Based on the following thought tree, provide a coherent response to the initial prompt: '{initial_prompt}'\n\n"
        response_prompt += self._format_thought_tree(thoughts)

        messages = [
            {"role": "system", "content": """You are an AI assistant that synthesizes complex thought trees into coherent responses. 
            Your output should follow this format:
            {
                "content": "A comprehensive response to the prompt...",
                "key_points": [
                    "Key point 1",
                    "Key point 2",
                    "Key point 3"
                ],
                "thought_process": "A brief explanation of how the thought tree led to this response..."
            }"""},
            {"role": "user", "content": response_prompt}
        ]

        return self.structured_response_generator.generate(messages, response_schema)

    def reflect(self, thoughts: List[Dict[str, Any]], response: Dict[str, Any], thinking_time: float) -> Dict[str, Any]:
        reflection_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A reflection on the thought process and response"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the reflection"},
                "areas_for_improvement": {"type": "array", "items": {"type": "string"}, "description": "Areas where the thinking process could be improved"},
                "confidence_level": {"type": "number", "description": "A score from 0 to 1 indicating confidence in the final response"}
            },
            "required": ["content", "key_points", "areas_for_improvement", "confidence_level"]
        }

        reflection_prompt = f"""Reflect on the following thought process and response. Consider the quality, depth, and breadth of the thoughts, the coherence of the response, and areas for improvement.

Thought Tree:
{self._format_thought_tree(thoughts)}

Response:
{json.dumps(response, indent=2)}

Thinking time: {thinking_time:.2f} seconds

Provide a detailed reflection, including key points, areas for improvement, and a confidence level in the final response."""

        messages = [
            {"role": "system", "content": """You are an AI assistant that reflects on thought processes and responses to improve future performance. 
            Your output should follow this format:
            {
                "content": "A detailed reflection on the thought process and response...",
                "key_points": [
                    "Key point 1",
                    "Key point 2",
                    "Key point 3"
                ],
                "areas_for_improvement": [
                    "Area for improvement 1",
                    "Area for improvement 2"
                ],
                "confidence_level": 0.85
            }"""},
            {"role": "user", "content": reflection_prompt}
        ]

        return self.structured_response_generator.generate(messages, reflection_schema)

    def _format_thought_tree(self, thoughts: List[Dict[str, Any]], indent: int = 0) -> str:
        formatted_tree = ""
        for i, thought in enumerate(thoughts, 1):
            formatted_tree += f"{'  ' * indent}Thought {i}:\n"
            formatted_tree += f"{'  ' * (indent + 1)}Content: {thought['content'][:200]}...\n"
            formatted_tree += f"{'  ' * (indent + 1)}Key Points: {', '.join(thought['key_points'])}\n"
            if thought['sub_thoughts']:
                formatted_tree += f"{'  ' * (indent + 1)}Sub-thoughts:\n"
                formatted_tree += self._format_thought_tree(thought['sub_thoughts'], indent + 2)
            formatted_tree += f"{'  ' * (indent + 1)}Continue Thinking: {thought['continue_thinking']}\n"
            formatted_tree += f"{'  ' * (indent + 1)}Reasoning: {thought['reasoning']}\n\n"
        return formatted_tree