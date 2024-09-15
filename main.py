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
from structured_response_generator import StructuredResponseGenerator
from thought_generator import ThoughtGenerator
from response_analyzer import ResponseAnalyzer
from final_response_generator import FinalResponseGenerator
from web_research_and_scraper import WebResearchAndScraper

# ANSI escape codes for colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_separator():
    print("\n" + "="*50 + "\n")

def print_colored(text, color):
    print(f"{color}{text}{Colors.ENDC}")

if __name__ == "__main__":
    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    structured_response_generator = StructuredResponseGenerator(client) # this is the model used to generate the responses. 
    web_researcher = WebResearchAndScraper(structured_response_generator)
    thought_generator = ThoughtGenerator(structured_response_generator, web_researcher) # this is the model used to generate the thoughts.
    response_analyzer = ResponseAnalyzer(structured_response_generator) # this is the model used to analyze the responses.
    final_response_generator = FinalResponseGenerator(structured_response_generator) # this is the model used to generate the final responses.

    initial_prompt = input("Enter the initial prompt: ") # this is the initial prompt for the entire process.

    try:
        print("Initiating web research...")
        print_separator()
        research_summary = web_researcher.conduct_research(initial_prompt)
        print("Web research completed.")
        print(f"Research summary: {research_summary['summary']}")
        print("Key points from research:")
        for point in research_summary['key_points']:
            print(f"- {point}")
        print_separator()

        print("Initiating thought generation process...")
        print_separator()
        
        thoughts, thinking_time = thought_generator.generate_thoughts(initial_prompt, research_summary)
        
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
            print(f"Conduct Research: {'Yes' if thought['conduct_research'] else 'No'}")
            print(f"Reasoning: {thought['reasoning']}")
            print("-" * 30)
        
        print_separator()
        print("Generating comprehensive response...")
        response = thought_generator.generate_response(thoughts, initial_prompt)
        print("\nComprehensive Response:")
        print(f"Content: {response['content']}")
        print("Key Points:")
        for point in response['key_points']:
            print(f"- {point}")
        print(f"Reasoning: {response['reasoning']}")
        
        print_separator()
        print("Reflecting on the thought process...")
        reflection = thought_generator.reflect(thoughts, response, thinking_time)
        print("\nReflection:")
        print(f"Content: {reflection['content']}")
        print("Key Points:")
        for point in reflection['key_points']:
            print(f"- {point}")
        print("\nAreas for Improvement:")
        for area in reflection['areas_for_improvement']:
            print(f"- {area}")
        print(f"\nConfidence Level: {reflection['confidence_level']:.2f}")
        print(f"\nMeta-Cognition: {reflection['meta_cognition']}")

        print_separator()
        print_colored("Generating final responses...", Colors.HEADER)
        final_responses = final_response_generator.generate_final_responses(thoughts, reflection, initial_prompt, research_summary)
        for i, response in enumerate(final_responses, 1):
            print_colored(f"\nFinal Response {i}:", Colors.BLUE)
            print(f"Content: {response['content']}")
            print_colored("Key Points:", Colors.CYAN)
            for point in response['key_points']:
                print(f"- {point}")
            print(f"Reasoning: {response['reasoning']}")
            print(f"Confidence Score: {response['confidence_score']:.2f}")
            print("-" * 30)
        
        print_separator()
        print_colored("Choosing the best final response...", Colors.HEADER)
        best_response_choice = final_response_generator.choose_best_response(final_responses)
        best_response = final_responses[best_response_choice['chosen_response'] - 1]
        
        print_separator()
        print_colored("Generating meta-analysis...", Colors.HEADER)
        meta_analysis = final_response_generator.generate_meta_analysis(final_responses, best_response_choice)
        print_colored("\nMeta-Analysis:", Colors.YELLOW)
        print(f"Overall Quality: {meta_analysis['overall_quality']}")
        print_colored("Key Insights:", Colors.CYAN)
        for insight in meta_analysis['key_insights']:
            print(f"- {insight}")
        print_colored("Areas for Improvement:", Colors.CYAN)
        for area in meta_analysis['areas_for_improvement']:
            print(f"- {area}")
        print(f"Confidence Analysis: {meta_analysis['confidence_analysis']}")

        # Color-coded final response as the last output
        print_separator()
        print_colored("FINAL RESPONSE:", Colors.BOLD + Colors.GREEN)
        print_colored(f"Content: {best_response['content']}", Colors.GREEN)
        print_colored("Key Points:", Colors.CYAN)
        for point in best_response['key_points']:
            print_colored(f"- {point}", Colors.CYAN)
        print_colored(f"Reasoning: {best_response['reasoning']}", Colors.YELLOW)
        print_colored(f"Confidence Score: {best_response['confidence_score']:.2f}", Colors.YELLOW)
        print_colored(f"\nReasoning for choice: {best_response_choice['reasoning']}", Colors.YELLOW)

    except Exception as e:
        print_colored(f"An error occurred: {e}", Colors.RED)
        print("Debug information:")
        print(f"Thoughts: {thoughts}")
        print(f"Initial prompt: {initial_prompt}")