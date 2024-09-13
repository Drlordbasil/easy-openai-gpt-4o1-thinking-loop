import json
import time
import random

class ThoughtGenerator:
    def __init__(self, response_generator):
        self.response_generator = response_generator

    def generate_thoughts(self, initial_prompt, max_iterations=10, max_time=60):
        thoughts = []
        messages = [
            {"role": "system", "content": """You are an AI assistant capable of deep, iterative thinking. Generate comprehensive and structured thoughts on the given topic. 
            Your output should strictly follow the JSON schema provided. When you believe you have explored the topic sufficiently, set 'continue_thinking' to false.
            
            Here's an example of a valid thought:
            {
                "content": "AGI could revolutionize healthcare by enabling personalized medicine and accelerating drug discovery. This could lead to longer lifespans and improved quality of life. However, it also raises concerns about privacy and data security in healthcare.",
                "key_points": [
                    "Personalized medicine",
                    "Accelerated drug discovery",
                    "Potential for longer lifespans",
                    "Improved quality of life",
                    "Privacy concerns in healthcare",
                    "Data security issues"
                ],
                "continue_thinking": true
            }
            
            Ensure your response strictly adheres to this format."""},
            {"role": "user", "content": initial_prompt}
        ]

        thought_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A detailed exploration of the topic"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "A list of key points from the content"},
                "continue_thinking": {"type": "boolean", "description": "Whether to continue the thought process"}
            },
            "required": ["content", "key_points", "continue_thinking"]
        }

        start_time = time.time()
        for i in range(max_iterations):
            thought = self.response_generator(messages, thought_schema)
            thoughts.append(thought)
            
            if not thought['continue_thinking']:
                break
            
            messages.append({"role": "assistant", "content": json.dumps(thought)})
            messages.append({"role": "user", "content": f"Thought {i+1} complete. Continue exploring this topic. Delve deeper or consider new angles. Remember to set 'continue_thinking' to false when you believe you've explored the topic sufficiently."})
            
            if time.time() - start_time > max_time:
                break

        thinking_time = time.time() - start_time
        return thoughts, thinking_time

    def generate_response(self, thoughts, initial_prompt):
        response_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A coherent response to the initial prompt"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "A list of key points from the response"}
            },
            "required": ["content", "key_points"]
        }

        response_prompt = f"Based on the following thoughts, provide a coherent response to the initial prompt: '{initial_prompt}'\n\n"
        for i, thought in enumerate(thoughts, 1):
            response_prompt += f"Thought {i}:\n{thought['content'][:200]}...\n\n"

        messages = [
            {"role": "system", "content": """You are an AI assistant that synthesizes thoughts into coherent responses. 
            Your output should follow this format:
            {
                "content": "A comprehensive response to the prompt...",
                "key_points": [
                    "Key point 1",
                    "Key point 2",
                    "Key point 3"
                ]
            }"""},
            {"role": "user", "content": response_prompt}
        ]

        return self.response_generator(messages, response_schema)

    def reflect(self, thoughts, response, thinking_time):
        reflection_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A reflection on the thought process and response"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key insights from the reflection"}
            },
            "required": ["content", "key_points"]
        }

        reflection_prompt = f"Reflect on the thought process that led to the following response. Consider the quality of the thoughts, the coherence of the response, and areas for improvement. Thinking time: {thinking_time:.2f} seconds.\n\nResponse:\n{response['content']}\n\nThoughts:\n"
        for i, thought in enumerate(thoughts, 1):
            reflection_prompt += f"Thought {i}:\n{thought['content'][:200]}...\n\n"

        messages = [
            {"role": "system", "content": """You are an AI assistant that reflects on thought processes and responses to improve future performance. 
            Your output should follow this format:
            {
                "content": "A detailed reflection on the thought process and response...",
                "key_points": [
                    "Key insight 1",
                    "Key insight 2",
                    "Key insight 3"
                ]
            }"""},
            {"role": "user", "content": reflection_prompt}
        ]

        return self.response_generator(messages, reflection_schema)

    def final_thought(self, reflection):
        final_thought_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A final thought on the topic, incorporating insights from the reflection"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the final thought"}
            },
            "required": ["content", "key_points"]
        }

        final_thought_prompt = f"Based on the following reflection, provide a final thought on the topic:\n\n{reflection['content']}"

        messages = [
            {"role": "system", "content": """You are an AI assistant that provides insightful final thoughts on a topic. 
            Your output should follow this format:
            {
                "content": "A comprehensive final thought on the topic...",
                "key_points": [
                    "Key point 1",
                    "Key point 2",
                    "Key point 3"
                ]
            }"""},
            {"role": "user", "content": final_thought_prompt}
        ]

        return self.response_generator(messages, final_thought_schema)

    def generate_final_responses(self, thoughts, reflection, initial_prompt, num_responses=3):
        final_response_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "A comprehensive final response to the initial prompt"},
                "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the final response"}
            },
            "required": ["content", "key_points"]
        }

        final_responses = []
        for i in range(num_responses):
            final_response_prompt = f"""Based on the following information, provide a comprehensive final response to the initial prompt: '{initial_prompt}'

Reflection:
{reflection['content']}

Key Insights:
{', '.join(reflection['key_points'])}

Generate a unique and insightful response that incorporates the lessons learned from the thought process and reflection.
"""

            messages = [
                {"role": "system", "content": f"""You are an AI assistant that provides comprehensive final responses on a topic. 
                Your output should follow this format:
                {{
                    "content": "A comprehensive final response to the prompt...",
                    "key_points": [
                        "Key point 1",
                        "Key point 2",
                        "Key point 3"
                    ]
                }}
                This is final response attempt {i+1} out of {num_responses}. Ensure each response is unique and insightful."""},
                {"role": "user", "content": final_response_prompt}
            ]

            final_responses.append(self.response_generator(messages, final_response_schema))

        return final_responses

    def choose_best_response(self, final_responses):
        best_response_schema = {
            "type": "object",
            "properties": {
                "chosen_response": {"type": "integer", "description": "The index of the chosen best response (1, 2, or 3)"}
            },
            "required": ["chosen_response"]
        }

        choice_prompt = f"""Analyze the following final responses and choose the best one based on its comprehensiveness, insight, and overall quality:

{json.dumps(final_responses, indent=2)}

Choose the best response and return only the index of the chosen response (1, 2, or 3).
"""

        messages = [
            {"role": "system", "content": """You are an AI assistant tasked with choosing the best final response from multiple options.
            Your output should follow this format:
            {
                "chosen_response": 1
            }"""},
            {"role": "user", "content": choice_prompt}
        ]

        choice = self.response_generator(messages, best_response_schema)
        return choice
