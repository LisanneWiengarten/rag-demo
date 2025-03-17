# Setup
- In the root directory, create a folder named `data` and copy all relevant pdfs there. For simplicity, input pdfs are always expected at this location
- Create a virtual environment for this service e.g. `conda create -n rag python=3.11` and activate it: `conda 
activate rag`
- Install the requirements: `pip install -r requirements.txt`
- Download spacy model: `python -m spacy download de_core_news_lg`
- Set your environment variables either in the terminal or in the [.env file](deployment/.env) (
  easiest is to copy the [example](deployment/.env.example), rename it and change the values.
- finally, run the script `python main.py --path path/to/output_dir --create` to create a new index at the given 
  path or `python main.py --path path/to/output_dir --load` to load an existing index from the given path


# Further TODOs
- Tests
- Logging
- Versioning
- Deployment
- Better user interface (e.g. allow user to type in questions)
- Better error handling
- Better prompts (for OCR and answer generation)
- Better data cleaning (e.g. remove all headers and footers)
- Better chunking (e.g. consider paragraph numbers)
- Tweak answers (e.g. more elaborate, include sources)
- Better data storage (e.g. save separate pages)
- More flexible LLM/OCR usage (e.g. easy option to use different vision model)
- More flexible methods (accept/return more basic types instead of nested dictionaries)
- Don't hardcode directories/file names (e.g. make it at least a constant)
