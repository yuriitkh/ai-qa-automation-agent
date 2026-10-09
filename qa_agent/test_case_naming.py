"""Names extracted from supplied content; never infer product behavior."""
import re


def content_title(text: str, *, maximum: int = 200) -> str:
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:(?:step|крок)\s*)?\d+(?:[.)\-:]\s*|\s+)", "", line, flags=re.I)
        line = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        words = re.findall(r"[^\W_]+(?:[’'-][^\W_]+)*", line, re.UNICODE)
        unique = []
        for word in words:
            if not unique or unique[-1].casefold() != word.casefold():
                unique.append(word)
        if not unique or all(word.casefold() in {"test", "draft", "new", "step", "tbd", "тест", "крок"} or word.isdecimal() for word in unique):
            continue
        if len(unique) > 10:
            unique = [word for word in unique if word.casefold() not in {"the", "a", "an", "please"}]
        if unique and unique[0][0].isdigit() and not unique[0].isdecimal():
            unique = [*unique[1:], unique[0]]
        title = " ".join(unique[:10])
        while len(title) > maximum and " " in title:
            title = title.rsplit(" ", 1)[0]
        if title and not title[0].isdigit():
            return title[:maximum]
    return ""


def fallback_summary(scenario: str, step_descriptions=()) -> str:
    for text in (scenario, *step_descriptions):
        title = content_title(text)
        if len(title.split()) >= 2:
            return title
    raise ValueError("Enter a Summary or describe the scenario so a meaningful Summary can be derived.")


def step_display_name(description: str) -> str:
    return content_title(description, maximum=80) or "Unfinished step"
