from bs4 import BeautifulSoup
from typing import List
from models import CodeLine

def parse_code_with_lines(html_code: str, class_table: dict[str, list[str]], selected_components: list[str]) -> List[CodeLine]:
    lines = html_code.strip().split("\n")
    results = []

    # 선택된 컴포넌트에서 수평 클래스 후보 수집
    horizontal_candidates = {
        h for comp in selected_components
        for h in class_table.get(comp, [])
    }

    for i, line in enumerate(lines, 1):
        el = BeautifulSoup(line, "html.parser").find()
        if el is None or line.strip().startswith("</"):
            continue

        tag = el.name
        class_list = el.get("class", [])

        available_horizontal = sorted([
            h for h in horizontal_candidates if h not in class_list
        ])

        results.append(CodeLine(
            line=i,
            tag=tag,
            classes=class_list,
            available_horizontal=available_horizontal
        ))

    return results
