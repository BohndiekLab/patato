import re

with open("paper.md") as file:
    text = file.read()

pattern = r"\[(@[A-Za-z0-9]+(?:;\s*@?[A-Za-z0-9]+)*)\]"

results = re.findall(pattern, text)

references = set()

for x in results:
    for y in x.split(";"):
        if "@" in y:
            y = y.replace("@", "")
        references.add(y.strip())


def simple_parse_bib(read):
    entries = []
    with open("paper.bib") as file:
        indent_level = 0
        in_entry = False
        current_entry = ""
        current_entry_title = ""
        while text := file.readline():
            indent_level += text.count("{")
            indent_level -= text.count("}")
            if text[0] == "@":
                assert indent_level == 1
                in_entry = True
                current_entry = text
                current_entry_title = text.split("{")[-1].split(",")[0]
            elif in_entry and indent_level == 0:
                current_entry += text
                if current_entry_title in read:
                    entries.append(current_entry)
                in_entry = False
            elif in_entry:
                current_entry += text
    return "\n".join(entries)


assert simple_parse_bib(references) == simple_parse_bib(simple_parse_bib(references))

with open("bibliography.bib", "w") as file:
    file.write(simple_parse_bib(references))
