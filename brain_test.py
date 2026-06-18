import ollama

print("Waking up the AI... please wait a few seconds.")

# We are sending a message to the phi3 model
response = ollama.chat(model='phi3', messages=[
    {'role': 'user', 'content': 'Say hello to the future top GitHub developer! Keep it short and encouraging.'},
])

# Print what the AI says back to us
print("\n--- AI RESPONSE ---")
print(response['message']['content'])
print("-------------------")