# StanleyAI Chatbot

## Setup

1. Create required directories:
   ```bash
   mkdir files data templates
   ```

2. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

3. Add your Groq API key to `.env`:
   ```bash
   GROQ_API_KEY=your_api_key_here
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Initialize the database:
   ```bash
   python templates/create_documents_db.py
   ```

6. Run the chatbot:
   ```bash
   python chatbot_groq_web.py
   ```

## Troubleshooting

### API Key Issues
If you see "GROQ_API_KEY environment variable not set":
1. Check that your `.env` file exists
2. Verify the API key is correctly formatted
3. Try setting it directly in your environment:
   ```bash
   set GROQ_API_KEY=your_api_key_here  # Windows
   export GROQ_API_KEY=your_api_key_here  # Linux/Mac
   ```

If you see "no such table: documents":
1. Make sure you've added documents to the `files` directory
2. Run `python templates/create_documents_db.py`
3. Verify `documents.db` was created

## Security Note
Never commit the `.env` file containing your API keys. The `.gitignore` file is configured to prevent this.
