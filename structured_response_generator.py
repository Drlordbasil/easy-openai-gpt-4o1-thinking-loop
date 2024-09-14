import json
import time
import openai

class StructuredResponseGenerator:
    def __init__(self, client, model="llama-3.1-70b-versatile"):
        self.client = client
        self.model = model

    def generate(self, messages, output_schema, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [
                        {"role": "system", "content": f"Respond with a JSON object that follows this schema: {json.dumps(output_schema)}"}
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                parsed_content = json.loads(content)
                
                # Validate the response against the schema
                if self._validate_schema(parsed_content, output_schema):
                    return parsed_content
                else:
                    raise ValueError("Response does not match the required schema")
                
            except (openai.BadRequestError, json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Error occurred: {e}. Retrying...")
                
                # Add feedback to the messages
                messages.append({"role": "system", "content": f"The previous response was invalid. Please ensure your response strictly adheres to the following JSON schema: {json.dumps(output_schema)}"})
                
                time.sleep(1)

    def _validate_schema(self, content, schema):
        required_properties = schema.get('properties', {}).keys()
        return all(prop in content for prop in required_properties)