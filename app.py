from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import traceback
import os

app = Flask(__name__)
CORS(app)

# Get token from environment variable (more secure) or use hardcoded value
# IMPORTANT: Replace 'YOUR_NEW_TOKEN_HERE' with your actual new token from Hugging Face
HF_TOKEN = os.environ.get('HUGGINGFACE_API_KEY')

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
        
        # Use free inference API models
        models_to_try = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/flan-t5-large",
            "gpt2-large",
            "distilgpt2"
        ]
        
        for model in models_to_try:
            try:
                print(f"Trying model: {model}")
                
                # Use the standard Inference API endpoint
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 250,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "do_sample": True
                    }
                }
                
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        generated_text = result.get('generated_text', str(result))
                    else:
                        generated_text = str(result)
                    
                    # Clean up the output (remove the input prompt if it's repeated)
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    print(f"Success with model: {model}")
                    return jsonify({
                        'generated_text': generated_text,
                        'model_used': model
                    })
                
                elif response.status_code == 503:
                    print(f"Model {model} is loading, trying next...")
                    continue
                    
                else:
                    print(f"Error response: {response.text}")
                    
            except Exception as model_error:
                print(f"Error with model {model}: {str(model_error)}")
                continue
        
        # If all models failed
        return jsonify({
            'error': 'All models are currently unavailable. Please try again in a moment.'
        }), 503
        
    except Exception as e:
        error_msg = f'Exception: {str(e)}\n{traceback.format_exc()}'
        print(error_msg)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/debug-token', methods=['GET'])
def debug_token():
    token = os.environ.get('HUGGINGFACE_API_KEY')
    if token:
        masked = f"{token[:7]}...{token[-4:]}" if len(token) > 11 else "TOO_SHORT"
        return jsonify({
            'token_found': True,
            'token_preview': masked,
            'token_length': len(token),
            'starts_with_hf': token.startswith('hf_')
        })
    else:
        return jsonify({
            'token_found': False,
            'message': 'HUGGINGFACE_API_KEY not set in environment'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)




