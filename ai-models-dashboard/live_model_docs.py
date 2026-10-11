"""Fetch selected public model documentation on demand; extract conservative facts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_BYTES = 6_000_000
TIMEOUT = 10

class DocumentationParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.main_blocks, self.article_blocks, self.buffer = [], [], [], []
        self.skip = self.main = self.article = 0

    def flush(self):
        value = " ".join(" ".join(self.buffer).split())
        self.buffer = []
        if value:
            self.blocks.append(value)
            if self.main:
                self.main_blocks.append(value)
            if self.article:
                self.article_blocks.append(value)

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "header", "aside", "noscript"}:
            self.skip += 1
        if self.skip:
            return
        if tag == "main":
            self.flush()
            self.main += 1
        if tag == "article":
            self.flush()
            self.article += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "div", "section", "br"}:
            self.flush()
        if tag in {"td", "th"}:
            self.buffer.append(" | ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header", "aside", "noscript"}:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "div", "section", "main", "article"}:
            self.flush()
        if tag == "main":
            self.main = max(0, self.main - 1)
        if tag == "article":
            self.article = max(0, self.article - 1)

    def handle_data(self, text):
        if not self.skip:
            self.buffer.append(text)

def document_lines(body):
    if re.search(r"<(?:html|main|article|body)\b", body[:10000], re.I):
        parser = DocumentationParser()
        parser.feed(body)
        parser.flush()
        lines = parser.article_blocks or parser.main_blocks or parser.blocks
    else:
        body = re.sub(r"\x60{3}.*?\x60{3}", "", body, flags=re.S)
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
        body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
        body = re.sub(r"<[^>]+>", " ", body)
        lines = body.splitlines()
    result = []
    for line in lines:
        line = re.sub(r"[\x60*_#]", "", line)
        line = " ".join(line.split()).strip(" |")
        if line and line not in result:
            result.append(line)
    return result

def fetch_document(url, opener=urlopen):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Documentation must use a public HTTPS URL.")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("A public model-documentation URL is required.")
    request = Request(url, headers={
        "User-Agent": "ModelComparisonDashboard/1.0",
        "Accept": "text/markdown,text/plain,text/html",
        "Cache-Control": "no-cache",
    })
    with opener(request, timeout=TIMEOUT) as response:
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Documentation is too large to read.")
        content_type = response.headers.get("Content-Type", "")
        if not any(value in content_type for value in ("text/", "application/xhtml", "application/json")):
            raise ValueError("This URL did not return readable model documentation.")
        encoding = response.headers.get_content_charset() or "utf-8"
        body = data.decode(encoding, errors="replace")
        final_url = response.geturl()
    lines = document_lines(body)
    if len(" ".join(lines)) < 150 or any(
        term in " ".join(lines[:8]).lower()
        for term in ("just a moment", "verify you are human", "access denied")
    ):
        raise ValueError("The documentation site did not return readable content.")
    return {"url": final_url, "lines": lines}

def source_urls(row, profile):
    urls = list(profile.get("documentation_urls", []))
    if not urls:
        url = profile.get("url")
        if url and "pricing" not in url:
            urls = [url]
        urls += [row["source"]]
    return list(dict.fromkeys(urls))[:2]

def preferred_url(url):
    parsed = urlparse(url)
    if parsed.hostname == "huggingface.co" and "/raw/" not in parsed.path:
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 2:
            return f"https://huggingface.co/{parts[0]}/{parts[1]}/raw/main/README.md"
    if parsed.hostname == "developers.openai.com" and "/api/docs/models/" in parsed.path and not parsed.path.endswith(".md"):
        return url.split("?")[0].rstrip("/") + ".md"
    return url

def relevant_lines(lines, row, url=None):
    if any(word in (url or row["source"]) for word in ("changelog", "/updates/")):
        date = row["date"]
        from datetime import datetime as dt
        human = dt.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y").replace(" 0", " ")
        start = next((i for i, line in enumerate(lines) if date in line or human in line), None)
        if start is None:
            return []
        stop = next((i for i in range(start + 1, len(lines)) if re.search(
            r"\b20\d\d-\d\d-\d\d\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, 20\d\d\b",
            lines[i])), min(start + 45, len(lines)))
        return lines[start:stop]
    return lines

def matching(lines, pattern):
    return [line for line in lines if re.search(pattern, line, re.I)]

def status(lines, term):
    section = ""
    for i, line in enumerate(lines):
        if re.fullmatch(r"(?:supported|unsupported) features", line, re.I):
            section = line.lower()
            continue
        if line.lower() in {"supported tools", "snapshots", "rate limits", "endpoints"}:
            section = ""
        compact_term = term.replace(" ", "[-_ ]*")
        if section and re.fullmatch(r"-?\s*(?:" + compact_term + r")", line, re.I):
            return "Not supported" if section.startswith("unsupported") else "Supported"
        if not re.search(term, line, re.I) or len(line) > 450:
            continue
        value = line
        if re.fullmatch(term, line, re.I) and i + 1 < len(lines):
            value += " " + lines[i + 1]
        if re.search(r"not supported|unsupported|not available|does not support|no support", value, re.I):
            return "Not supported"
        if re.search(r"\bsupported\b|\bsupports\b|\bavailable\b|\benables?\b", value, re.I):
            return "Supported"
    return None

def extract_facts(lines):
    facts = {}
    whole = "\n".join(lines)
    tasks = []
    for label, pattern in [
        ("coding", r"\bcoding\b|code generation"),
        ("reasoning", r"\breasoning\b"),
        ("summaries", r"summari[sz]"),
        ("classification", r"classification"),
        ("computer use", r"computer use|browser use"),
        ("agents", r"agentic|agent workflows"),
        ("multilingual tasks", r"multilingual"),
        ("visual understanding", r"visual reasoning|image understanding"),
    ]:
        if re.search(pattern, whole, re.I):
            tasks.append(label)
    if tasks:
        facts["Performance evidence / release focus"] = "Tasks described: " + ", ".join(tasks) + "."
    context_values = []
    for line in matching(lines, r"context(?: window| length)?|max(?:imum)? (?:output|sequence)"):
        if len(line) > 450 or re.search(r"bench|evaluat|baseline|harness|temperature|temp=|in-house", line, re.I):
            continue
        qualifier = ("Planned hosted context" if re.search(r"hosted", line, re.I) and re.search(r"coming soon|will be", line, re.I)
                     else "Native context" if re.search(r"nativ", line, re.I) else "Context")
        for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?\s*[kKmM]?)\s*(?:tokens?\s*)?(?:context window|context length|max output tokens|max sequence length)", line, re.I):
            label = "Maximum output" if re.search(r"max output", match.group(0), re.I) else qualifier
            context_values.append(label + ": " + match.group(1).strip() + " tokens")
        for match in re.finditer(r"(?:context(?: window| length)?|max(?:imum)? sequence length)\s*(?:of|:|=|\||is|up to)?\s*(\d[\d,]*(?:\.\d+)?\s*[kKmM]?)", line, re.I):
            context_values.append(qualifier + ": " + match.group(1).strip() + " tokens")
        if re.search(r"extensib|extend", line, re.I):
            extension = re.search(r"(?:extensible|extended|extend).*?up to\s*(\d[\d,]*(?:\.\d+)?\s*[kKmM]?)", line, re.I)
            if extension:
                context_values.append("With extension: " + extension.group(1).strip() + " tokens")
    if context_values:
        def token_count(value):
            number = re.search(r":\s*(\d[\d,]*(?:\.\d+)?)\s*([kKmM]?)", value)
            if not number:
                return None
            multiplier = {"k": 1000, "m": 1000000}.get(number.group(2).lower(), 1)
            return int(float(number.group(1).replace(",", "")) * multiplier)
        qualified = {token_count(value) for value in context_values
                     if value.startswith(("Native context:", "Planned hosted context:", "With extension:"))}
        context_values = [value for value in context_values
                          if not value.startswith("Context:") or token_count(value) not in qualified]
        facts["Context window"] = "; ".join(dict.fromkeys(context_values)) + "."
    modalities = []
    inputs, outputs, unsupported = set(), set(), set()
    for line in lines:
        for label, target in [("Input modalities", inputs), ("Output modalities", outputs), ("Unsupported modalities", unsupported)]:
            if re.search(label + r"\s*:", line, re.I):
                target.update(re.findall(r"\b(?:text|image|audio|video)\b", line.lower()))
    if inputs or outputs or unsupported:
        for label in ["Text", "Image", "Audio", "Video"]:
            name = label.lower()
            direction = ("input and output" if name in inputs and name in outputs else
                         "input" if name in inputs else "output" if name in outputs else
                         "not supported" if name in unsupported else None)
            if direction:
                modalities.append(label + ": " + direction)
    for label in ["Text", "Image", "Audio", "Video"]:
        if any(value.startswith(label + ":") for value in modalities):
            continue
        for i, line in enumerate(lines):
            if re.fullmatch(label, line, re.I) and i + 1 < len(lines):
                following = lines[i + 1]
                if re.search(r"input|output|not supported", following, re.I):
                    modalities.append(label + ": " + following[:70].rstrip("."))
                    break
            if len(line) < 180 and re.search(rf"\b{label}\b.*\b(?:input|output)\b", line, re.I):
                direction = "input and output" if re.search("input and output", line, re.I) else (
                    "input" if re.search("input", line, re.I) else "output")
                negative = re.search("not supported|does not support", line, re.I)
                modalities.append(label + ": " + ("not supported" if negative else direction))
                break
    if modalities:
        facts["Input / output types"] = "; ".join(modalities) + "."
    security = []
    for label, pattern in [
        ("data residency", r"data residency"),
        ("zero data retention", r"zero.data.retention"),
        ("safety safeguards", r"\bsafeguards\b"),
        ("safety evaluation", r"safety (?:evaluations?|testing)|system card"),
        ("misuse restrictions", r"restrict.*(?:cyber|biolog|harm)|block.*penetration"),
    ]:
        if re.search(pattern, whole, re.I):
            security.append(label)
    if security:
        facts["Security and data handling"] = "Documentation covers: " + ", ".join(security) + "."
    platforms = []
    for label, pattern in [
        ("API", r"\bAPI\b"),
        ("Amazon Web Services", r"Amazon Web Services|\bAWS\b|\bBedrock\b"),
        ("Google Cloud / Vertex AI", r"Google Cloud|Vertex AI"),
        ("Microsoft Azure", r"Microsoft Azure|\bAzure\b"),
        ("vLLM", r"\bvLLM\b"),
        ("SGLang", r"\bSGLang\b"),
        ("Transformers", r"\btransformers\b"),
        ("llama.cpp", r"llama\.cpp"),
        ("Ollama", r"\bOllama\b"),
    ]:
        if re.search(pattern, whole, re.I):
            platforms.append(label)
    if platforms:
        facts["Deployment and operations"] = "Platforms / serving tools documented: " + ", ".join(platforms) + "."
    customization = []
    fine_tuning = status(lines, r"fine[- ]?tuning")
    if fine_tuning:
        customization.append("Fine-tuning: " + fine_tuning.lower())
    if re.search(r"\bLoRA\b|\bQLoRA\b", whole):
        customization.append("LoRA adapters are discussed")
    if re.search(r"adjustable effort|reasoning[._ ]?effort|effort setting", whole, re.I):
        customization.append("Reasoning effort can be adjusted")
    if customization:
        facts["Customization"] = "; ".join(customization) + "."
    license_lines = matching(lines, r"\blicen[sc]e\b|commercial use")
    licenses = []
    for label, pattern in [
        ("Apache 2.0", r"apache[- ]?(?:license[- ]?)?(?:2\.0|2)"),
        ("MIT", r"\bMIT\b"),
        ("Gemma terms", r"gemma terms|gemma license agreement|gemma terms of use"),
        ("Llama community license", r"llama.*community.*licen[sc]e|llama.*licen[sc]e"),
    ]:
        if any(re.search(pattern, line, re.I) for line in license_lines):
            licenses.append(label)
    if licenses:
        facts["License / commercial use"] = "License named: " + ", ".join(licenses) + "."
    integration = []
    for label, pattern in [
        ("Function calling", r"function calling"),
        ("Structured outputs", r"structured outputs?"),
        ("Streaming", r"streaming"),
        ("Tool use", r"tool (?:use|calling)"),
    ]:
        value = status(lines, pattern)
        if value:
            integration.append(label + ": " + value.lower())
    for label, pattern in [("Responses API", r"Responses API|v1/responses"),
                           ("Chat Completions API", r"Chat Completions|v1/chat/completions"),
                           ("Python SDK", r"Python.*SDK|SDK.*Python"),
                           ("TypeScript SDK", r"TypeScript.*SDK|SDK.*TypeScript")]:
        if re.search(pattern, whole, re.I):
            integration.append(label)
    if integration:
        facts["Integration and support"] = "; ".join(integration) + "."
    return facts

def load_model(row, profile, fetcher=fetch_document):
    facts, sources, failures, documents = {}, {}, [], []
    for url in source_urls(row, profile):
        target = preferred_url(url)
        try:
            try:
                document = fetcher(target)
            except Exception:
                if target == url:
                    raise
                document = fetcher(url)
            lines = relevant_lines(document["lines"], row, document["url"])
            if not lines:
                raise ValueError("The selected release was not found in this documentation.")
            document_facts = extract_facts(lines)
            documents.append(document["url"])
            for factor, value in document_facts.items():
                if factor not in facts:
                    facts[factor] = value
                    sources[factor] = document["url"]
        except Exception as error:
            failures.append({"url": url, "reason": str(error)})
    return {"title": row["title"], "facts": facts, "sources": sources,
            "documents": documents, "failures": failures}

def similar(left, right):
    def normalized(value):
        return re.sub(r"[^a-z0-9.]+", " ", value.casefold()).strip()
    a, b = normalized(left), normalized(right)
    # Extraction uses canonical wording: equivalent facts have the same text.
    # A fuzzy prose score could hide an extra platform, a token limit or "not".
    return a == b

def differing_factors(models):
    factors = set(models[0]["facts"]) | set(models[1]["facts"])
    differing, shared, incomplete = {}, [], []
    for factor in sorted(factors):
        values = [model["facts"].get(factor) for model in models]
        if None in values:
            incomplete.append(factor)
        elif similar(*values):
            shared.append(factor)
        else:
            differing[factor] = values
    return differing, shared, incomplete

def compare_live_documents(rows, profiles, fetcher=fetch_document):
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(load_model, row, profiles.get(row["title"], {}), fetcher) for row in rows]
        models = [future.result() for future in futures]
    differing, shared, incomplete = differing_factors(models)
    return {"models": models, "differing": differing, "shared": shared,
            "incomplete": incomplete,
            "fetched_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M:%S UTC")}

