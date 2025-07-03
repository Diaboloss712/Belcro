import json, pathlib, os, re, functools, requests
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_KEY=os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
META = pathlib.Path("data/meta.json")

def _meta_version() -> str | None:
    if META.is_file():
        try:
            return json.loads(META.read_text())["doc_version"]
        except Exception:
            pass
    return None

_DOC_FROM_META   = _meta_version()
_DOC_FROM_ENV    = os.getenv("BOOTSTRAP_DOC_VERSION")

@functools.lru_cache(maxsize=1)
def detect_latest_version() -> str:
    xml = requests.get("https://getbootstrap.com/sitemap-0.xml", timeout=10).text
    m = re.search(r"/docs/(\d+\.\d+)/", xml)
    return m.group(1) if m else "5.3"

DOC_VERSION = "0.0"

QUESTIONS = [
    "In a Bootstrap 5 card body, what class combination keeps a checkbox and its label on one horizontal line?",
    "How do I replace navbar-brand text with a logo image in Bootstrap 5?",
    "Which data attribute prevents a Bootstrap 5 carousel from pausing on hover?",
    "Which value of data-bs-backdrop makes a modal static so clicks outside do not dismiss it?",
    "What utility class makes an alert background the same color as the .bg-warning class?"
]

COMPONENTS: list[str] | None = None
