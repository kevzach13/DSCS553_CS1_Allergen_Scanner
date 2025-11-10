# sitecustomize.py
# Auto-imported by Python if it’s on sys.path. We patch gradio_client safely.

import os

# 1) kill proxies so Gradio's localhost check doesn't route through a proxy
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"]  = "127.0.0.1,localhost"

# 2) patch the JSON-schema helper functions that crash when schema is a bool
try:
    import gradio_client.utils as _gcu

    _old_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "any"
        return _old_get_type(schema)
    _gcu.get_type = _safe_get_type

    _old_js2py = _gcu._json_schema_to_python_type
    def _safe_js2py(schema, defs=None):
        if not isinstance(schema, dict):
            return "any"
        ap = schema.get("additionalProperties")
        if isinstance(ap, bool):
            schema = dict(schema)
            schema["additionalProperties"] = {}
        return _old_js2py(schema, defs)
    _gcu._json_schema_to_python_type = _safe_js2py
except Exception:
    pass  # don't break startup if client utils move

# 3) stop Gradio from building the API schema entirely (belt & suspenders)
try:
    import gradio.blocks as _g_blocks
    def _noop_api_info(self):  # returns nothing instead of walking the schema
        return {}
    _g_blocks.Blocks.get_api_info = _noop_api_info
except Exception:
    pass

# 4) disable analytics if your version honors this
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
