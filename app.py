from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import traceback
import os

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get('HUGGINGFACE_API_KEY')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        # Use the Hugging Face router endpoint (OpenAI-compatible)
        api_url = "https://router.huggingface.co/v1/chat/completions"

        # Free/public models that generally work on the router
        models_to_try = [
            "gpt2",
            "distilgpt2",
            "facebook/opt-350m",
            "tiiuae/falcon-1b"
        ]

        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                }

                print(f"Trying model: {model}")
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                print(f"Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result["choices"][0]["message"]["content"]
                    print(f"Success with model: {model}")
                    return jsonify({
                        "generated_text": generated_text,
                        "model_used": model
                    })

                else:
                    print(f"Error response from {model}: {response.text}")

            except Exception as model_error:
                print(f"Error with model {model}: {str(model_error)}")
                continue

        return jsonify({
            'error': 'All models are currently unavailable. Please try again later.'
        }), 503

    except Exception as e:
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
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
