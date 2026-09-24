"""Streamlit page: upload a food photo, get a Food-11 prediction.

The inference service's address comes from INFERENCE_URL so the same image runs
inside Compose (`http://inference:8000`) and standalone against a locally run
API (`http://127.0.0.1:8000`) with no code change.
"""

import os

import requests
import streamlit as st

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 30

st.set_page_config(page_title="Food-11 classifier")
st.title("Food-11 classifier")
st.caption(f"Inference service: `{INFERENCE_URL}`")


def service_health():
    """Ask the inference service what it is serving, or None if unreachable."""
    try:
        response = requests.get(f"{INFERENCE_URL}/health", timeout=5)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


health = service_health()
if health is None:
    st.error(
        "Cannot reach the inference service. If the stack was just started, "
        "give it a moment; otherwise check `docker compose logs inference`."
    )
else:
    st.caption(f"Model: `{health.get('model_uri')}`")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, width=300)

    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    try:
        response = requests.post(
            f"{INFERENCE_URL}/predict", files=files, timeout=TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        st.error(f"Request to the inference service failed: {exc}")
    else:
        if response.ok:
            result = response.json()
            st.success(
                f"**{result['category']}** — {result['confidence']:.1%} confidence"
            )

            probabilities = result.get("probabilities", {})
            if probabilities:
                ranked = sorted(
                    probabilities.items(), key=lambda item: item[1], reverse=True
                )
                st.subheader("All categories")
                for name, probability in ranked:
                    st.progress(
                        min(float(probability), 1.0),
                        text=f"{name} — {probability:.1%}",
                    )
        else:
            st.error(
                f"Inference service returned {response.status_code}: {response.text}"
            )