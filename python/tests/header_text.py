"""C++ header text handling for the defaults parser: comments, braces, struct bodies."""

import re


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def matching_brace(text, opening):
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("unbalanced braces")


def struct_body(text, struct):
    match = re.search(r"\bstruct\s+" + re.escape(struct) + r"\b\s*\{", text)
    assert match, f"no struct {struct} in the header"
    opening = match.end() - 1
    return text[opening + 1 : matching_brace(text, opening)]


def split_top_level(text, separator):
    parts = []
    depth = 0
    current = ""
    for character in text:
        if character in "{<(":
            depth += 1
        elif character in "}>)":
            depth -= 1
        if character == separator and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += character
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def declared_constants(text):
    pattern = r"static\s+constexpr\s+[\w:<>,\s]*?\b(\w+)\s*=\s*([^;]+);"
    return {name: value.strip() for name, value in re.findall(pattern, text)}


def brace_initialized_members(body):
    """Top-level `Type name{init};` declarations, in declaration order.

    A member function body and a requires-clause are both a brace run that is
    not followed by a semicolon, which is what distinguishes them from a member
    initializer here. Splitting on semicolons alone cannot do it, because a
    function definition carries none.
    """
    members = []
    index = 0
    head_start = 0
    while index < len(body):
        character = body[index]
        if character == "{":
            closing = matching_brace(body, index)
            after = closing + 1
            while after < len(body) and body[after].isspace():
                after += 1
            if after < len(body) and body[after] == ";":
                head = body[head_start:index].strip()
                name = re.search(r"(\w+)\s*$", head)
                if name and not name.group(1).endswith("_"):
                    members.append((name.group(1), body[index + 1 : closing].strip(), head))
                index = after + 1
            else:
                index = closing + 1
            head_start = index
            continue
        if character == ";":
            index += 1
            head_start = index
            continue
        index += 1
    return members
