# Setup
- Copy relevant pdf files into data/pdfs. For simplicity, input pdfs are always expected at this location
- Create a virtual environment for this service e.g. `conda create -n rag python=3.11` and activate it: `conda 
activate rag`
- Install the requirements: `pip install -r requirements.txt`
- Set your environment variables either in the terminal or in the [.env file](deployment/.env) (
  easiest is to copy the [example](deployment/.env.example), rename it and change the values.
- finally, run the script


# Further TODOs
- Tests
- Logging
- Versioning
- Deployment
- Better user interface
- Better error handling
- Better prompts
- Better data cleaning
- Better chunking
- Better answers (more elaborate, include sources)
- Better data storage (e.g. save separate pages)
- More flexible LLM/OCR usage
- More flexible methods (accept/return more basic types instead of nested dictionaries)