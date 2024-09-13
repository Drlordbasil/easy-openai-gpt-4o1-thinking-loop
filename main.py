### Welcome to the true thinking AI ### 
# this model can think and learn from its own thoughts and responses.
# it can also analyze its own thoughts and responses.
# it can also reflect on its own thoughts and responses.
# it can also generate a final thought on a topic.
# it uses groq's API to connect to the openai API.
# you can change the model, the API key, and the base URL to whatever you want. 
# the model used here is llama-3.1-70b-versatile, which is a large language model.
# the API key is the key for the groq API. you can get one by signing up at https://console.groq.com/ or change with the base URL of your OpenAI-compatible API.            
# the base URL is the base URL for the groq API. you can get one by signing up at https://console.groq.com/ or change with the base URL of your OpenAI-compatible API.
# the output schema is the schema for the response. you can change this to whatever you want.
# the thinking time is the time it takes for the model to think and learn from its own thoughts and responses. you can change this to whatever you want.
# the thinking depth is the depth of the thoughts. you can change this to whatever you want.
# the number of iterations is the number of iterations the model will go through. you can change this to whatever you want.
# the thought depth is the depth of the thoughts. you can change this to whatever you want. 
# the response analyzer is the response analyzer. you can change this to whatever you want.

import os
import openai
import json
import time
from thought_generator import ThoughtGenerator

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", # You can replace this with the base URL of your OpenAI-compatible API of your choosing.
    api_key=os.environ.get("GROQ_API_KEY") # This is the API key for the Groq API. You can get one by signing up at https://console.groq.com/ or change with the base URL of your OpenAI-compatible API.
) 

def generate_structured_response(messages, output_schema, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile", # This is the model we are using. You can change this to any other model that is compatible with OpenAI's API.
                messages=messages + [
                    {"role": "system", "content": f"Respond with a JSON object that follows this schema: {json.dumps(output_schema)}"}
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except (openai.BadRequestError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(1)

def print_separator():
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    initial_prompt = "Explore the potential long-term impacts of artificial general intelligence on human society."
    
    thought_generator = ThoughtGenerator(generate_structured_response)
    
    try:
        print("Initiating thought generation process...")
        print_separator()
        
        thoughts, thinking_time = thought_generator.generate_thoughts(initial_prompt)
        
        print(f"Thought generation completed in {thinking_time:.2f} seconds.")
        print(f"Generated {len(thoughts)} thoughts.")
        print_separator()
        
        print("Thought Stream:")
        for i, thought in enumerate(thoughts, 1):
            print(f"\nThought {i}:")
            print(f"Content: {thought['content']}")
            print("Key Points:")
            for point in thought['key_points']:
                print(f"- {point}")
            print(f"Continue Thinking: {'Yes' if thought['continue_thinking'] else 'No'}")
            print("-" * 30)
        
        print_separator()
        print("Generating comprehensive response...")
        response = thought_generator.generate_response(thoughts, initial_prompt)
        print("\nComprehensive Response:")
        print(f"Content: {response['content']}")
        print("Key Points:")
        for point in response['key_points']:
            print(f"- {point}")
        
        print_separator()
        print("Reflecting on the thought process...")
        reflection = thought_generator.reflect(thoughts, response, thinking_time)
        print("\nReflection:")
        print(f"Content: {reflection['content']}")
        print("Key Insights:")
        for insight in reflection['key_points']:
            print(f"- {insight}")
        
        print_separator()
        print("Generating final responses...")
        final_responses = thought_generator.generate_final_responses(thoughts, reflection, initial_prompt)
        for i, response in enumerate(final_responses, 1):
            print(f"\nFinal Response {i}:")
            print(f"Content: {response['content']}")
            print("Key Points:")
            for point in response['key_points']:
                print(f"- {point}")
            print("-" * 30)
        
        print_separator()
        print("Choosing the best final response...")
        best_response_choice = thought_generator.choose_best_response(final_responses)
        best_response = final_responses[best_response_choice['chosen_response'] - 1]
        
        print("\nBest Final Response:")
        print(f"Content: {best_response['content']}")
        print("Key Points:")
        for point in best_response['key_points']:
            print(f"- {point}")
    except Exception as e:
        print(f"An error occurred: {e}")
