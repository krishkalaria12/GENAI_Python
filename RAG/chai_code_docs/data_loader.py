from langchain_community.document_loaders import WebBaseLoader

class DataLoader:
    """Handles loading documents from various sources."""
    
    def __init__(self):
        self.chai_code_urls = [
            "https://docs.chaicode.com/youtube/getting-started/",
            "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
            "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
            "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
            "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
            "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
            "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
            "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
            "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
            "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
            "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
            "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
            "https://docs.chaicode.com/youtube/chai-aur-git/github/",
            "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
            "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
            "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
            "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
            "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
            "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
            "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
            "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
            "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
            "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
            "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
            "https://docs.chaicode.com/youtube/chai-aur-django/models/",
            "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
            "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
            "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
            "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
            "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"
        ]
    
    def load_chai_code_docs(self, urls: list = None) -> list:
        """Load documents from ChaiCode documentation URLs."""
        if urls is None:
            urls = self.chai_code_urls
        
        print(f"Loading documents from {len(urls)} URLs...")
        
        loader = WebBaseLoader(urls)
        loader.requests_kwargs = {
            'verify': False,  # Disable SSL verification
        }
        
        try:
            docs = loader.load()
            print(f"Successfully loaded {len(docs)} documents")
            return docs
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise
    
    def load_custom_urls(self, urls: list) -> list:
        """Load documents from custom URLs."""
        return self.load_chai_code_docs(urls) 