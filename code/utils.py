
class Utils:
    def __init__(self, API_KEY, OPEN_API_KEY):
        self.pinecone_api_key = API_KEY
        self.openai_api_key = OPEN_API_KEY

    def create_index_name(self, index_name, OPENAI_API_KEY):
        openai_key = ''
        openai_key = OPENAI_API_KEY #os.getenv("OPENAI_API_KEY")
        return f'{index_name}-{openai_key[-36:].lower().replace("_", "-")}'
    

