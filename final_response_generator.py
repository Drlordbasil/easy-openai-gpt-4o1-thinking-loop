import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class FinalResponseGenerator:
    def __init__(self, structured_response_generator):
        self.structured_response_generator = structured_response_generator

    def generate_final_responses(self, thoughts: List[Dict[str, Any]], reflection: Dict[str, Any], initial_prompt: str, research_summary: Dict[str, Any], num_responses: int = 3) -> List[Dict[str, Any]]:
        final_response_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A comprehensive final response to the initial prompt"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the final response"},
                "reasoning": {"type": "string", "description": "Reasoning behind the final response"},
                "confidence_score": {"type": "number", "description": "Confidence score for the response (0-1)"}
            },
            "required": ["content", "key_points", "reasoning", "confidence_score"]
        }

        with ThreadPoolExecutor(max_workers=num_responses) as executor:
            future_to_response = {executor.submit(self._generate_single_response, thoughts, reflection, initial_prompt, research_summary, final_response_schema, i): i for i in range(num_responses)}
            final_responses = []
            for future in as_completed(future_to_response):
                response = future.result()
                if response:
                    final_responses.append(response)

        return final_responses

    def _generate_single_response(self, thoughts: List[Dict[str, Any]], reflection: Dict[str, Any], initial_prompt: str, research_summary: Dict[str, Any], final_response_schema: Dict[str, Any], response_index: int) -> Dict[str, Any]:
        final_response_prompt = self._create_final_response_prompt(thoughts, reflection, initial_prompt, research_summary, response_index)

        messages = [
            {"role": "system", "content": "You are an AI assistant that provides comprehensive final responses on a topic. Your response should be unique, insightful, and incorporate lessons learned from the thought process, reflection, and research."},
            {"role": "user", "content": final_response_prompt}
        ]

        try:
            response = self.structured_response_generator.generate(messages, final_response_schema)
            return response
        except Exception as e:
            print(f"Error generating final response {response_index + 1}: {e}")
            return None

    def _create_final_response_prompt(self, thoughts: List[Dict[str, Any]], reflection: Dict[str, Any], initial_prompt: str, research_summary: Dict[str, Any], response_index: int) -> str:
        prompt = f"""Based on the following information, provide a comprehensive final response to the initial prompt: '{initial_prompt}'

Research Summary:
{research_summary['summary']}

Key Research Points:
{self._format_list(research_summary['key_points'])}

Thought Process:
{self._format_thoughts(thoughts)}

Reflection:
{reflection['content']}

Reflection Key Points:
{self._format_list(reflection['key_points'])}

Generate a unique and insightful response (attempt {response_index + 1}) that incorporates the lessons learned from the thought process, reflection, and research. Ensure to include key points in your response, provide reasoning for your conclusions, and assign a confidence score to your response.
"""
        return prompt

    def _format_thoughts(self, thoughts: List[Dict[str, Any]]) -> str:
        formatted_thoughts = ""
        for i, thought in enumerate(thoughts, 1):
            formatted_thoughts += f"Thought {i}:\n"
            formatted_thoughts += f"Content: {thought['content'][:200]}...\n"
            formatted_thoughts += f"Key Points: {', '.join(thought['key_points'])}\n\n"
        return formatted_thoughts

    def _format_list(self, items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def choose_best_response(self, final_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        best_response_schema = {
            "type": "object",
            "properties": {
                "chosen_response": {"type": "integer", "description": "The index of the chosen best response (1-based)"},
                "reasoning": {"type": "string", "description": "Reasoning for choosing this response"}
            },
            "required": ["chosen_response", "reasoning"]
        }

        choice_prompt = f"""Analyze the following final responses and choose the best one based on its comprehensiveness, insight, reasoning, and overall quality:

{json.dumps(final_responses, indent=2)}

Choose the best response and provide reasoning for your choice. Return the index of the chosen response (1-based) and your reasoning.
"""

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with choosing the best final response from multiple options. Provide a thorough analysis and justification for your choice."},
            {"role": "user", "content": choice_prompt}
        ]

        choice = self.structured_response_generator.generate(messages, best_response_schema)
        return choice

    def generate_meta_analysis(self, final_responses: List[Dict[str, Any]], best_response: Dict[str, Any]) -> Dict[str, Any]:
        meta_analysis_schema = {
            "type": "object",
            "properties": {
                "overall_quality": {"type": "string", "description": "Assessment of the overall quality of the responses"},
                "key_insights": {"type": "array", "items": {"type": "string"}, "description": "Key insights from all responses"},
                "areas_for_improvement": {"type": "array", "items": {"type": "string"}, "description": "Areas where the responses could be improved"},
                "confidence_analysis": {"type": "string", "description": "Analysis of the confidence scores across responses"}
            },
            "required": ["overall_quality", "key_insights", "areas_for_improvement", "confidence_analysis"]
        }

        meta_analysis_prompt = f"""Perform a meta-analysis of the following final responses and the chosen best response:

Final Responses:
{json.dumps(final_responses, indent=2)}

Chosen Best Response:
{json.dumps(best_response, indent=2)}

Provide an assessment of the overall quality of the responses, key insights from all responses, areas for improvement, and an analysis of the confidence scores across responses.
"""

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with performing a meta-analysis of multiple responses to a complex question. Provide a thorough and insightful analysis."},
            {"role": "user", "content": meta_analysis_prompt}
        ]

        meta_analysis = self.structured_response_generator.generate(messages, meta_analysis_schema)
        return meta_analysis