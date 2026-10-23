import httpx
import json

from .conftest import FakeBackend


def sse_events(text: str) -> list[tuple[str | None, dict]]:
    events, name = [], None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and line[5:].strip() != "[DONE]":
            events.append((name, json.loads(line[5:])))
            name = None
    return events


# ------------------------------------------------------------------ OpenAI

def test_openai_local(make_client):
    client, local, cloud = make_client()
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "resume esto"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hola desde local"
    assert r.headers["x-pair-route"].startswith("local")
    assert local.requests[0]["model"] == "qwen-local" and not cloud.requests


def test_openai_stream_cloud_sends_completion_tokens_param(make_client):
    client, _, cloud = make_client()
    r = client.post("/v1/chat/completions", json={
        "model": "x", "stream": True, "max_tokens": 50,
        "messages": [{"role": "user", "content": "@frontera piensa"}]})
    events = sse_events(r.text)
    text = "".join(e["choices"][0]["delta"].get("content", "") for _, e in events if e.get("choices"))
    assert text.strip() == "hola desde cloud"
    assert r.text.rstrip().endswith("[DONE]")
    assert cloud.requests[0]["max_completion_tokens"] == 50 and "max_tokens" not in cloud.requests[0]
    assert cloud.requests[0]["messages"][0]["content"] == "piensa"


# ------------------------------------------------------------------ Anthropic

def test_anthropic_tool_roundtrip_input(make_client):
    client, local, _ = make_client()
    r = client.post("/v1/messages", json={
        "model": "claude-x", "max_tokens": 100, "system": [{"type": "text", "text": "eres útil"}],
        "tools": [{"name": "ls", "description": "lista", "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {"role": "user", "content": "resume la carpeta"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]},
        ]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message" and body["model"] == "claude-x"
    assert body["content"][0]["text"] == "hola desde local"
    sent = local.requests[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool"]
    assert sent["messages"][3]["tool_call_id"] == "toolu_1"
    assert sent["tools"][0]["function"]["name"] == "ls"


def test_anthropic_stream_tool_call(make_client):
    cloud = FakeBackend("cloud", "gpt-test", tool_call={"name": "ls", "args": {"path": "/tmp"}})
    client, _, _ = make_client(cloud=cloud, mode="calidad")
    r = client.post("/v1/messages", json={
        "model": "claude-x", "max_tokens": 100, "stream": True,
        "tools": [{"name": "ls", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "mira /tmp"}]})
    events = sse_events(r.text)
    names = [n for n, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    start = next(e for n, e in events if n == "content_block_start")
    assert start["content_block"]["type"] == "tool_use" and start["content_block"]["name"] == "ls"
    partial = "".join(e["delta"]["partial_json"] for n, e in events if n == "content_block_delta")
    assert json.loads(partial) == {"path": "/tmp"}
    delta = next(e for n, e in events if n == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_anthropic_count_tokens(make_client):
    client, _, _ = make_client()
    r = client.post("/v1/messages/count_tokens", json={"model": "c", "messages": [{"role": "user", "content": "x" * 400}]})
    assert r.json()["input_tokens"] == 100


# ------------------------------------------------------------------ Gemini

def test_gemini_function_call(make_client):
    local = FakeBackend("local", "qwen-local", tool_call={"name": "tiempo", "args": {"ciudad": "Alicante"}})
    client, local, _ = make_client(local=local)
    r = client.post("/v1beta/models/gemini-x:generateContent", json={
        "systemInstruction": {"parts": [{"text": "sé breve"}]},
        "contents": [{"role": "user", "parts": [{"text": "¿qué tiempo hace?"}]}],
        "tools": [{"functionDeclarations": [{"name": "tiempo", "parameters": {
            "type": "OBJECT", "properties": {"ciudad": {"type": "STRING"}}}}]}],
        "generationConfig": {"maxOutputTokens": 64}})
    assert r.status_code == 200
    part = r.json()["candidates"][0]["content"]["parts"][0]
    assert part["functionCall"]["name"] == "tiempo" and part["functionCall"]["args"] == {"ciudad": "Alicante"}
    sent = local.requests[0]
    assert sent["tools"][0]["function"]["parameters"]["properties"]["ciudad"]["type"] == "string"
    assert sent["max_tokens"] == 64


def test_gemini_stream(make_client):
    client, _, _ = make_client()
    r = client.post("/v1beta/models/gemini-x:streamGenerateContent?alt=sse",
                    json={"contents": [{"role": "user", "parts": [{"text": "hola"}]}]})
    events = sse_events(r.text)
    text = "".join(p.get("text", "") for _, e in events for p in e["candidates"][0]["content"]["parts"])
    assert text.strip() == "hola desde local"
    assert events[-1][1]["candidates"][0]["finishReason"] == "STOP"


# ------------------------------------------------------------------ fallback, privacidad y registro

def test_local_down_falls_back_to_cloud(make_client):
    client, _, cloud = make_client(local=FakeBackend("local", "q", down=True))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "resume"}]})
    assert r.status_code == 200 and cloud.requests
    assert "fallback" in r.headers["x-pair-route"]


def test_sensitive_with_local_down_fails_instead_of_leaking(make_client):
    client, _, cloud = make_client(local=FakeBackend("local", "q", down=True))
    r = client.post("/v1/messages", json={"model": "c", "max_tokens": 10,
                                          "messages": [{"role": "user", "content": "DNI 12345678Z"}]})
    assert r.status_code == 502 and r.json()["type"] == "error"
    assert not cloud.requests


def test_stats_and_bench(make_client, tmp_path):
    client, _, _ = make_client()
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "resume esto"}]})
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "@frontera hola"}]})
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "DNI 12345678Z"}]})
    s = client.get("/pair/stats").json()
    assert s["requests"] == 3 and s["local"] == 2 and s["sensitive"] == 1
    # 100 tok entrada a 1 USD/M + 50 salida a 10 USD/M = 0.0006 por petición
    assert s["all_cloud_usd"] == 0.0018 and s["cost_usd"] == 0.0006
    store = client.app.state.pair.store
    n = store.export_bench(str(tmp_path / "bench.csv"))
    assert n == 2  # la sensible no se guarda


def test_describe_request_structure():
    from pair.app import describe
    out = describe({"model": "x", "stream": True, "messages": [
        {"role": "system", "content": "eres útil"},
        {"role": "user", "content": [{"type": "text", "text": "hola"}, {"type": "file", "file": {"content": "abc"}}]}]})
    assert "system: texto(9)" in out
    assert "file['file']" in out and "otros campos: model, stream" in out


# ------------------------------------------------------------------ imágenes

def test_image_local_by_default(make_client):
    client, _, cloud = make_client()
    r = client.post("/v1/images/generations", json={"model": "pair-image", "prompt": "un faro al atardecer",
                                                    "size": "1536x1024"})
    assert r.status_code == 200 and r.headers["x-pair-route"].startswith("local")
    assert r.json()["data"][0]["b64_json"] == "UE5HTE9DQUw="     # b"PNGLOCAL"
    assert client.flux.requests[0] == {"prompt": "un faro al atardecer", "referencias": [], "mejorar": True, "ancho": 1536, "alto": 1024}
    assert not cloud.requests


def test_image_cloud_tag_and_cost(make_client):
    client, _, cloud = make_client()
    r = client.post("/v1/images/generations", json={"prompt": "@frontera un faro", "n": 1})
    assert r.headers["x-pair-route"].startswith("cloud")
    assert cloud.requests[0] == {"prompt": "un faro", "n": 1, "model": "gpt-image-test"}
    s = client.get("/pair/stats").json()
    assert s["cost_usd"] == 0.04


def test_image_falls_back_to_cloud_when_local_down(make_client):
    from .conftest import FakeFlux
    client, _, cloud = make_client(flux=FakeFlux(down=True))
    r = client.post("/v1/images/generations", json={"prompt": "un faro", "response_format": "url"})
    assert r.status_code == 200 and "fallback" in r.headers["x-pair-route"]
    assert r.json()["data"][0]["url"].startswith("data:image/png;base64,")


def test_models_include_image_model(make_client):
    client, _, _ = make_client()
    assert {m["id"] for m in client.get("/v1/models").json()["data"]} == {"pair", "pair-image"}


# ------------------------------------------------------------------ contexto real de LM Studio

def test_context_and_model_from_lmstudio(make_client):
    from .conftest import FakeBackend

    class LMStudio(FakeBackend):
        def handler(self, request):
            if request.url.path == "/api/v0/models":
                return httpx.Response(200, json={"data": [
                    {"id": "gemma", "type": "llm", "state": "not-loaded", "max_context_length": 262144},
                    {"id": "glm-4.7", "type": "llm", "state": "loaded", "loaded_context_length": 8192}]})
            return super().handler(request)

    client, local, _ = make_client(local=LMStudio("local", "ignorado"), mode="ahorro")
    # ~10.000 tokens no caben en 8192 * 0,9 → nube, aunque el modo sea ahorro
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x " * 20000}]})
    assert r.headers["x-pair-route"].startswith("cloud") and "supera el local" in r.headers["x-pair-route"]
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hola"}]})
    assert r.headers["x-pair-model"] == "glm-4.7" and local.requests[0]["model"] == "glm-4.7"


def test_local_reasoning_not_counted_as_savings(make_client):
    from .conftest import FakeBackend

    class Thinker(FakeBackend):
        def handler(self, request):
            r = super().handler(request)
            if request.url.path.endswith("/chat/completions"):
                body = r.json()
                body["usage"] = {"prompt_tokens": 100, "completion_tokens": 1050,
                                 "completion_tokens_details": {"reasoning_tokens": 1000}}
                return httpx.Response(200, json=body)
            return r

    client, _, _ = make_client(local=Thinker("local", "glm"))
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "resume esto"}]})
    s = client.get("/pair/stats").json()
    # 100 entrada a 1 USD/M + 50 visibles a 10 USD/M = 0.0006, no 0.0106
    assert s["all_cloud_usd"] == 0.0006


def test_image_from_chat_tag_stream(make_client):
    client, local, _ = make_client()
    r = client.post("/v1/chat/completions", json={"model": "pair", "stream": True, "messages": [
        {"role": "user", "content": "hola"}, {"role": "assistant", "content": "hola"},
        {"role": "user", "content": "@imagen un faro al atardecer"}]})
    events = sse_events(r.text)
    content = events[0][1]["choices"][0]["delta"]["content"]
    url = content.removeprefix("![imagen](").removesuffix(")")
    assert url.startswith("http://testserver/imagenes/") and url.endswith(".png")
    assert client.get(url).content == b"PNGLOCAL"
    assert client.flux.requests[0]["prompt"] == "un faro al atardecer"
    assert not local.requests and r.headers["x-pair-route"].startswith("local")


def test_image_from_chat_image_model(make_client):
    client, _, cloud = make_client()
    r = client.post("/v1/chat/completions", json={"model": "PAIR-image", "messages": [
        {"role": "user", "content": "@frontera un faro"}]})
    assert r.json()["choices"][0]["message"]["content"].startswith("![imagen](http://testserver/imagenes/")
    assert cloud.requests[0]["prompt"] == "un faro"


def test_generated_images_are_dropped_from_history(make_client):
    client, local, _ = make_client()
    big = "![imagen](data:image/png;base64," + "A" * 500000 + ")"
    client.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "@imagen un faro"}, {"role": "assistant", "content": big},
        {"role": "user", "content": "gracias"}]})
    assert local.requests[0]["messages"][1]["content"] == "[imagen generada]"


def test_embedded_images_dropped_from_any_message(make_client):
    client, local, _ = make_client()
    title_request = "Pon título a esta conversación: ![imagen](data:image/png;base64," + "A" * 500000 + ")"
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": title_request}]})
    assert r.headers["x-pair-route"].startswith("local")
    assert local.requests[0]["messages"][0]["content"].endswith("[imagen generada]")


def test_image_file_name_is_validated(make_client):
    client, _, _ = make_client()
    assert client.get("/imagenes/..%2Fpair.db").status_code == 404
