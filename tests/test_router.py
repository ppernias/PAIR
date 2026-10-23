from pair.router import CLOUD, LOCAL, decide

from .conftest import RULES


def req(text: str, **extra) -> dict:
    return {"messages": [{"role": "user", "content": text}], **extra}


def test_tag_local_is_removed():
    r = req("@local escribe un poema")
    d = decide(r, RULES, "calidad")
    assert d.target == LOCAL and "etiqueta" in d.reason
    assert r["messages"][0]["content"] == "escribe un poema"


def test_tag_cloud():
    assert decide(req("resume esto @frontera"), RULES, "ahorro").target == CLOUD


def test_sensitive_never_leaves_even_with_cloud_tag():
    d = decide(req("@frontera revisa el DNI 12345678Z"), RULES, "calidad")
    assert d.target == LOCAL and d.sensitive


def test_keywords():
    assert decide(req("Resume este párrafo"), RULES, "calidad").target == LOCAL
    assert decide(req("Diseña la arquitectura del sistema"), RULES, "ahorro").target == CLOUD


def test_keyword_needs_whole_word():
    # "resumen" está en la lista, pero "resúmenes" no debe coincidir: decide el modo
    d = decide(req("dame los resúmenes"), RULES, "calidad")
    assert d.target == CLOUD and d.reason == "modo calidad"


def test_images_rule():
    r = lambda: {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe la imagen"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]}
    # por defecto el modelo local ve imágenes; con images: true van a la nube
    assert decide(r(), RULES, "ahorro").target == LOCAL
    con_regla = {**RULES, "force_cloud_if": {**RULES["force_cloud_if"], "images": True}}
    assert decide(r(), con_regla, "ahorro").target == CLOUD


def test_context_too_big_goes_cloud():
    assert decide(req("resume " + "palabra " * 20000), RULES, "ahorro", local_max_context=16000).target == CLOUD


def test_modes():
    assert decide(req("hola"), RULES, "ahorro").target == LOCAL
    assert decide(req("hola"), RULES, "calidad").target == CLOUD
    assert decide(req("hola"), RULES, "equilibrado").target == LOCAL
    assert decide(req("texto " * 3000), RULES, "equilibrado").target == CLOUD


def test_non_text_parts_count_for_size_and_privacy():
    from pair.router import estimate_tokens, is_sensitive
    r = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "resume"},
        {"type": "file", "file": {"filename": "doc.pdf", "content": "palabra " * 400}}]}]}
    assert estimate_tokens(r) > 700
    r2 = {"messages": [{"role": "user", "content": [
        {"type": "file", "file": {"content": "el DNI es 12345678Z"}}]}]}
    assert is_sensitive(r2, RULES)


def test_base64_is_not_counted_as_text():
    from pair.router import estimate_tokens
    r = {"messages": [{"role": "user", "content": [
        {"type": "file", "file": {"file_data": "data:application/pdf;base64," + "A" * 40000}}]}]}
    assert estimate_tokens(r) < 10


def test_balanced_mode_ignores_history_and_system_prompt():
    r = {"messages": [
        {"role": "system", "content": "instrucciones " * 2000},
        {"role": "user", "content": "documento " * 3000},
        {"role": "assistant", "content": "respuesta " * 500},
        {"role": "user", "content": "gracias"},
    ]}
    d = decide(r, RULES, "equilibrado", local_max_context=128000)
    assert d.target == LOCAL and "último mensaje" in d.reason


def test_agent_round_uses_original_user_message():
    r = {"messages": [
        {"role": "user", "content": "qué hay en la carpeta"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
                                                               "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "a.txt\nb.txt"},
    ]}
    assert decide(r, RULES, "equilibrado").target == LOCAL
