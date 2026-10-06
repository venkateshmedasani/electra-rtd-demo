"""
ELECTRA live demo: "Beat the Discriminator"

Flow (matches Figure 2 of Clark et al., 2020):
  1. You (or the audience) pick a word in a sentence.
  2. The small ELECTRA GENERATOR proposes plausible replacement words
     for that position (this is the masked-language-model half).
  3. You pick a suggestion -- or the audience shouts out their own word
     to try to fool the model.
  4. The ELECTRA DISCRIMINATOR scores every token in the resulting
     sentence as "real" (green) or "replaced" (red).

Run with:  python app.py
Then open the printed http://127.0.0.1:7860 link in your browser.
No internet needed at runtime if you ran preload_models.py first.
"""

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, ElectraForMaskedLM, ElectraForPreTraining

GEN_NAME = "google/electra-small-generator"
DISC_NAME = "google/electra-small-discriminator"

print("Loading ELECTRA models (instant if preload_models.py was already run)...")
tokenizer = AutoTokenizer.from_pretrained(DISC_NAME)
# Models stay on CPU at load time -- ZeroGPU only attaches a GPU for the
# duration of an @spaces.GPU-decorated call, so we move each model to
# 'cuda' inside those functions, not here at module load.
gen_model = ElectraForMaskedLM.from_pretrained(GEN_NAME).eval()
disc_model = ElectraForPreTraining.from_pretrained(DISC_NAME).eval()
print("Models ready.\n")

EXAMPLE_SENTENCES = [
    "The chef cooked the meal in the kitchen",
    "The scientist published her results in a major journal",
    "He walked the dog around the block every morning",
]


def split_words(sentence: str):
    return sentence.strip().split()


@spaces.GPU
def get_suggestions(sentence, word_to_replace, top_k=5):
    """Step 1: mask the chosen word and ask the GENERATOR to fill it in."""
    words = split_words(sentence)
    if not word_to_replace or word_to_replace not in words:
        return gr.update(choices=[], value=None), "⚠️ That word isn't in the sentence above (check spelling/case, and avoid punctuation)."

    idx = words.index(word_to_replace)
    masked_words = words.copy()
    masked_words[idx] = tokenizer.mask_token
    masked_sentence = " ".join(masked_words)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_model.to(device)
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    mask_positions = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    if len(mask_positions) == 0:
        return gr.update(choices=[], value=None), "⚠️ Tokenization hiccup — try a simpler word."
    mask_pos = mask_positions[0]

    with torch.no_grad():
        logits = gen_model(**inputs).logits

    top_ids = torch.topk(logits[0, mask_pos, :], top_k).indices.cpu().tolist()
    candidates = [tokenizer.decode([t]).strip() for t in top_ids]
    candidates = [c for c in candidates if c]

    note = f"Generator's top {len(candidates)} guesses for the blank — pick one below, or let someone in the audience type their own."
    return gr.update(choices=candidates, value=candidates[0] if candidates else None), note


@spaces.GPU
def render_heatmap(sentence, word_to_replace, chosen_candidate, custom_word):
    """Step 2: substitute the word, run the DISCRIMINATOR, color every token."""
    words = split_words(sentence)
    if not word_to_replace or word_to_replace not in words:
        return "", "⚠️ Pick a valid word first (step 1)."

    replacement = (custom_word or "").strip() or chosen_candidate
    if not replacement:
        return "", "⚠️ Choose a suggestion or type your own word first."

    idx = words.index(word_to_replace)
    corrupted_words = words.copy()
    corrupted_words[idx] = replacement
    corrupted_sentence = " ".join(corrupted_words)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc_model.to(device)
    inputs = tokenizer(corrupted_sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = disc_model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    spans = []
    for tok, p in zip(tokens, probs):
        if tok in tokenizer.all_special_tokens:
            continue
        clean = tok.replace("##", "")
        r = int(255 * p)          # more "replaced" -> more red
        g = int(255 * (1 - p))    # more "real" -> more green
        spans.append(
            f"<span style='background-color:rgb({r},{g},90); "
            f"padding:8px 12px; margin:4px; border-radius:6px; "
            f"display:inline-block; font-size:24px; color:black;'>"
            f"{clean}<br><span style='font-size:13px;'>{p:.2f}</span></span>"
        )
    html = "<div style='line-height:2.8;'>" + " ".join(spans) + "</div>"
    caption = (
        f"Corrupted sentence fed to the discriminator: \"{corrupted_sentence}\"  "
        f"(green = model thinks 'real', red = model thinks 'replaced')"
    )
    return html, caption


with gr.Blocks(title="ELECTRA: Beat the Discriminator") as demo:
    gr.Markdown(
        "## ELECTRA — Replaced Token Detection, live\n"
        "Pick a word, get the generator's replacement suggestions, then see if the "
        "discriminator catches the fake."
    )

    sentence = gr.Textbox(label="1. Sentence", value=EXAMPLE_SENTENCES[0])
    word_to_replace = gr.Textbox(
        label="2. Word to replace (type it exactly as it appears — avoid the last word / punctuation)",
        value="chef",
    )
    suggest_btn = gr.Button("3. Get generator's suggestions", variant="primary")

    suggestion_note = gr.Markdown()
    candidates = gr.Radio(label="Generator's top candidates — pick one", choices=[])
    custom_word = gr.Textbox(label="...or type a word (e.g. from the audience) to try to fool it")

    run_btn = gr.Button("4. Run the discriminator", variant="primary")
    heatmap = gr.HTML()
    caption = gr.Markdown()

    suggest_btn.click(
        get_suggestions,
        inputs=[sentence, word_to_replace],
        outputs=[candidates, suggestion_note],
    )
    run_btn.click(
        render_heatmap,
        inputs=[sentence, word_to_replace, candidates, custom_word],
        outputs=[heatmap, caption],
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)