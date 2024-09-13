import os
import time
import random
from groq import Groq

def generate_groq_response(messages, model="llama3-8b-8192"):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content

def generate_thoughts(initial_prompt, min_time=5, max_time=30):
    messages = [
        {"role": "system", "content": "You are an AI assistant capable of deep, iterative thinking. Generate comprehensive thoughts and review them critically."},
        {"role": "user", "content": initial_prompt}
    ]
    
    start_time = time.time()
    elapsed_time = 0
    thoughts = []
    
    while elapsed_time < max_time:
        response = generate_groq_response(messages)
        thoughts.append(response)
        
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Continue this line of thinking. Expand on the previous thoughts and explore new angles."})
        
        elapsed_time = time.time() - start_time
        if elapsed_time >= min_time:
            break
        
        time.sleep(random.uniform(1, 3))  # Random pause between thoughts
    
    review_prompt = f"Review and critically analyze the following thoughts on the initial prompt '{initial_prompt}':\n\n"
    for i, thought in enumerate(thoughts, 1):
        review_prompt += f"Thought {i}: {thought}\n\n"
    review_prompt += "Based on this review, provide a final, comprehensive response that synthesizes and expands upon these thoughts."
    
    messages.append({"role": "user", "content": review_prompt})
    final_response = generate_groq_response(messages)
    
    return final_response, elapsed_time, thoughts

if __name__ == "__main__":
    initial_prompt = "Explore the potential long-term impacts of artificial general intelligence on human society."
    
    final_response, thinking_time, all_thoughts = generate_thoughts(initial_prompt)
    
    print(f"Thought for {thinking_time:.2f} seconds")
    print("\nThinking process:")
    for i, thought in enumerate(all_thoughts, 1):
        print(f"Thought {i}: {thought[:100]}...")  # Print first 100 characters of each thought
    print("\nFinal response:")
    print(final_response)
