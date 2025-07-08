import json, pathlib, os, re, functools, requests
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_KEY=os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
META = pathlib.Path("data/meta.json")
PINE_INDEX = "upstage-index"

def _meta_version() -> str | None:
    if META.is_file():
        try:
            return json.loads(META.read_text())["doc_version"]
        except Exception:
            pass
    return None

@functools.lru_cache(maxsize=1)
def detect_latest_version() -> str:
    xml = requests.get("https://getbootstrap.com/sitemap-0.xml", timeout=10).text
    m = re.search(r"/docs/(\d+\.\d+)/", xml)
    return m.group(1) if m else "5.3"

COMPONENTS: list[str] | None = None
