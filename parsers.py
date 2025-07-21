from bs4 import BeautifulSoup
from typing import List
from models import CodeLine

def parse_code_with_lines(html_code: str, class_table: dict[str, list[str]]) -> List[CodeLine]:
    lines = html_code.strip().split("\n")
    results = []

    for i, line in enumerate(lines, 1):
        el = BeautifulSoup(line, "html.parser").find()
        if el is None or line.strip().startswith("</"):
            continue

        tag = el.name
        class_list = el.get("class", [])
        tag_key = tag.lower()

        results.append(CodeLine(
            line=i,
            tag=tag,
            classes=class_list,
            available_horizontal=[
                cls for cls in class_table.get(tag_key, []) if cls not in class_list
            ]
        ))

    return results
