# Project 2 Writeup: Netflix Campaign Design Generator

## Situation
A creative agency needs rapid, high-quality visual concepts for Netflix-style digital marketing banners and posters. Traditional design workflows are slower for early campaign ideation.

## Task
Create a web platform that converts text prompts into bespoke marketing designs using OpenAI DALL·E and Gradio.

## Action
1. **Libraries:** Used `openai`, `gradio`, `requests`, `PIL`, and `io.BytesIO`.
2. **`generate_image` function:** Sends enhanced prompts to DALL·E 2, fetches the generated image URL, and returns a PIL image object.
3. **Gradio UI:** Built `gr.Interface` with a text prompt input, image size selector, and generated image output.
4. **Campaign examples:** Added sample prompts for thriller, romance, action, and documentary-style Netflix campaigns.

## Result
Designers can prototype campaign visuals in seconds from natural-language prompts. The platform supports iterative creative exploration and reduces time-to-concept for promotional assets.

## Key Architecture Visualizations
- Notebook with successful `generate_image` test output
- Gradio UI home screen
- 2–3 generated Netflix-style poster examples with different prompts
