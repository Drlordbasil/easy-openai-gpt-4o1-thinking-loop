import json

class FinalResponseGenerator:
    def __init__(self, structured_response_generator):
        self.structured_response_generator = structured_response_generator

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
            # Debug print
            print(f"Generating final response {i+1}")
            print(f"Reflection content: {reflection.get('content', 'No content')}")
            print(f"Reflection keys: {reflection.keys()}")

            final_response_prompt = f"""Based on the following information, provide a comprehensive final response to the initial prompt: '{initial_prompt}'

Reflection:
{reflection.get('content', 'No reflection content available')}

Key Points:
{', '.join(reflection.get('key_points', ['No key points available']))}

Generate a unique and insightful response that incorporates the lessons learned from the thought process and reflection. Ensure to include key points in your response.
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

            final_responses.append(self.structured_response_generator.generate(messages, final_response_schema))

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

        choice = self.structured_response_generator.generate(messages, best_response_schema)
        return choice