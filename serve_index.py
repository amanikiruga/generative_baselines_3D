"""Expose index.html (and its referenced assets) via a Gradio share link."""
import re
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent.resolve()
HTML_PATH = ROOT / "index_v4_n50_merged.html"
ASSET_PREFIX_TOKEN = "@@ASSETBASE@@"  # placeholder we substitute at load time

html = HTML_PATH.read_text()

# Whitelist every dir + every symlink target so Gradio's allowed_paths check passes.
allowed = {str(ROOT)}
for child in ROOT.iterdir():
    if child.is_symlink() or child.is_dir():
        allowed.add(str(child.resolve()))

# Rewrite static src=/href= referring to local files (e.g. eval_outputs/...) to
# Gradio's file-serving endpoint with the *resolved* absolute path.
def rewrite_static(match):
    attr, path = match.group(1), match.group(2)
    if path.startswith(("http://", "https://", "//", "#", "data:", "/gradio_api/")):
        return match.group(0)
    abs_path = (ROOT / path).resolve()
    return f'{attr}="/gradio_api/file={abs_path}"'

html = re.sub(r'(src|href)="([^"]+)"', rewrite_static, html)

# Patch the JS recipe builder to route relative asset paths through Gradio's file URL.
html = html.replace(
    "media.src = src;",
    "media.src = window.__ASSET_RESOLVE__(src);",
)

# Extract the <script>...</script> block(s) — gr.HTML inserts via innerHTML, which
# does NOT execute scripts. We move them into the <head> via gr.Blocks(head=...).
scripts = []
def grab_script(m):
    scripts.append(m.group(0))
    return ""
html = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", grab_script, html)

# eval_abs    = repr(str((ROOT / "eval_outputs").resolve()))
# eval_v2_abs = repr(str((ROOT / "eval_outputs_v2").resolve()))
eval_v4_merged_abs = repr(str((ROOT / "eval_outputs_v4_n50_merged").resolve()))
eval_v4_n50_abs    = repr(str((ROOT / "eval_outputs_v4_n50").resolve()))
eval_v4_array_abs  = repr(str((ROOT / "eval_outputs_v4_n50_array").resolve()))
eval_v3_abs        = repr(str((ROOT / "eval_outputs_v3").resolve()))
eval_v3_n50_abs    = repr(str((ROOT / "eval_outputs_v3_n50").resolve()))
asset_resolver = f"""
<script>
window.__ASSET_RESOLVE__ = (rel) => {{
  if (!rel || /^(https?:|data:|\\/)/.test(rel)) return rel;
  // Order matters: longer prefix first so eval_outputs_v4_n50_merged matches before eval_outputs_v4_n50.
  const map = {{
    "eval_outputs_v4_n50_merged": {eval_v4_merged_abs},
    "eval_outputs_v4_n50_array":  {eval_v4_array_abs},
    "eval_outputs_v4_n50":        {eval_v4_n50_abs},
    "eval_outputs_v3_n50":        {eval_v3_n50_abs},
    "eval_outputs_v3":            {eval_v3_abs}
  }};
  for (const k of Object.keys(map)) {{
    if (rel.startsWith(k + "/")) return "/gradio_api/file=" + map[k] + rel.slice(k.length);
  }}
  return rel;
}};
</script>
"""
head_extra = asset_resolver + "\n".join(scripts)

# Force light styling and override Gradio's dark/washed-out theme bleed.
override_css = """
html, body, .gradio-container, .gradio-container * { color: #111 !important; }
html, body, .gradio-container { background: #fff !important; }
.gradio-container { max-width: none !important; padding: 0 !important; }
.gradio-container .prose, .gradio-container .html { color: #111 !important; opacity: 1 !important; }
footer, .footer, .built-with { display: none !important; }
"""

with gr.Blocks(title="generative baselines", theme=gr.themes.Default(),
               css=override_css, head=head_extra) as demo:
    gr.HTML(html)

demo.launch(
    share=True,
    server_name="0.0.0.0",
    allowed_paths=list(allowed),
)
