from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from models import CodeLine

def parse_code_with_lines(
    html_code: str,
    horizontal_table: Dict[str, List[str]],
    selected_components: List[str],
    global_utilities: Optional[List[str]] = None
) -> List[CodeLine]:

    lines = html_code.strip().split("\n")
    results = []

    component_prefix_map = {}
    for comp_name, h_classes in horizontal_table.items():
        if h_classes:
            prefix = h_classes[0].split('-')[0]
            if prefix:
                component_prefix_map[prefix] = comp_name

    for i, line in enumerate(lines, 1):
        el = BeautifulSoup(line, "html.parser").find()
        if el is None or line.strip().startswith("</"):
            continue

        tag = el.name
        class_list = el.get("class", [])

        line_component = None
        for cls in class_list:
            prefix = cls.split('-')[0]
            if prefix in component_prefix_map:
                line_component = component_prefix_map[prefix]
                break

        all_available_horizontal = set()

        if line_component:
            all_available_horizontal.update(horizontal_table.get(line_component, []))

        if global_utilities:
            all_available_horizontal.update(global_utilities)

        current_classes_on_line = set(class_list)
        available_horizontal = sorted(list(all_available_horizontal - current_classes_on_line))

        results.append(CodeLine(
            line=i,
            tag=tag,
            classes=class_list,
            available_horizontal=available_horizontal
        ))

    return results
