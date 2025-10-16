from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import traceback

app = Flask(__name__)
CORS(app)

# PUT YOUR TOKEN HERE
HF_TOKEN = "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"  # Replace with your actual Hugging Face token

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        print(f"Received prompt: {prompt}")
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        headers = {
            'Authorization': f'Bearer {HF_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        # Try multiple models
        models_to_try = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "microsoft/Phi-3.5-mini-instruct"
        ]
        
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            print(f"Trying model: {model}")
            
            response = requests.post(
                'https://router.huggingface.co/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                print(f"Success with model: {model}")
                return jsonify({'generated_text': generated_text})
        
        # If all models failed
        return jsonify({'error': 'No models available'}), 500
        
    except Exception as e:
        error_msg = f'Exception: {str(e)}\n{traceback.format_exc()}'
        print(error_msg)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
