import json

class ResponseAnalyzer:
    def __init__(self, response_generator):
        self.response_generator = response_generator

    def analyze_thoughts(self, initial_prompt, all_thoughts, output_schema):
        analysis_prompt = f"""Analyze the following thoughts on the initial prompt: '{initial_prompt}'

Thoughts:
{self._format_thoughts(all_thoughts)}

Provide a comprehensive analysis of these thoughts, highlighting key trends, insights, and potential implications. 
Your response should follow the specified output format and ensure proper JSON formatting."""

        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing and synthesizing complex ideas. Ensure your output follows the specified JSON format and uses proper JSON formatting."},
            {"role": "user", "content": analysis_prompt}
        ]

        analysis = self.response_generator(messages, output_schema)
        analysis['type'] = 'analysis'
        return analysis

    def _format_thoughts(self, thoughts):
        formatted_thoughts = ""
        for i, thought in enumerate(thoughts, 1):
            formatted_thoughts += f"Thought {i} ({thought['type']}):\n"
            formatted_thoughts += f"Content: {thought['content'][:200]}...\n"
            formatted_thoughts += f"Key Points: {', '.join(thought['key_points'])}\n\n"
        return formatted_thoughts
