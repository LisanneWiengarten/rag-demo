# Setup
- Copy relevant pdf files into data/pdfs
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