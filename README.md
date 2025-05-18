# StanleyAI Chatbot

## Setup

1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Add your Groq API key to `.env`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the chatbot:
   ```bash
   python chatbot.py
   ```

## Security Note
Never commit the `.env` file containing your API keys. The `.gitignore` file is configured to prevent this.
