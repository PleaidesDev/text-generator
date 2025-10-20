from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import traceback

app = Flask(__name__)
CORS(app)

# Load Hugging Face token from Render environment variable
HF_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")

# Initialize OpenAI-compatible client for Hugging Face Router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        print(f"Received prompt: {prompt}")

        # Send request to Hugging Face Router using OpenAI-compatible API
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b:groq",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract text from the router response
        generated_text = completion.choices[0].message.content

        print(f"Success: Generated text from openai/gpt-oss-20b:groq")
        return jsonify({
            "generated_text": generated_text,
            "model_used": "openai/gpt-oss-20b:groq"
        })

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
