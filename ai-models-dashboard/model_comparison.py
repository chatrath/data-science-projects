"""Interactive, evidence-aware comparison of releases from the timeline CSV."""
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st
from live_model_docs import compare_live_documents, source_urls

FACTOR_HELP = {
    "Date / family": "When this model was released and the group of models it belongs to.",
    "Performance evidence / release focus": "Tasks and capabilities described in the documentation fetched for this comparison.",
    "Context window": "How much text or other information the model can consider at one time.",
    "Input / output types": "The types of content the model can receive and produce, such as text or images.",
    "Security and data handling": "How your information is stored, used and protected when using the model.",
    "Deployment and operations": "Where the model runs and what is needed to keep it working.",
    "Customization": "How you can adapt the model to your own tasks or information.",
    "License / commercial use": "The rules for using the model, including use in a business.",
    "Integration and support": "How the model connects to your tools and what help is available if something goes wrong.",
    "Model Documentation": "Link 1 opens the official README, model card or technical documentation; Link 2 opens the official model website.",
}

def comparison_html(table, documentation=None):
    """Escape dataset text and add click-to-open definitions to each factor."""
    display = table.rename_axis(index=None, columns="Factor").copy()
    display = display.map(lambda value: escape(str(value)))
    display.columns = [escape(str(name)) for name in display.columns]
    display.columns.name = "Factor"
    if documentation:
        display.loc["Model Documentation"] = [
            '<br>'.join(
                '<a href="' + escape(url, quote=True)
                + '" target="_blank" rel="noopener noreferrer">Link ' + str(index) + '</a>'
                for index, url in enumerate(documentation[title], 1)
            ) for title in table.columns
        ]
    display.index = [
        escape(factor) + '<details class="factor-help"><summary aria-label="Definition of '
        + escape(factor, quote=True) + '">ⓘ</summary><span>'
        + escape(FACTOR_HELP[factor]) + '</span></details>' for factor in display.index
    ]
    return display.to_html(classes="comparison-factors", border=0, escape=False)

def model_options(timeline, kind):
    return timeline[timeline["kind"] == kind].sort_values(
        "date", ascending=False, kind="stable"
    ).drop_duplicates("title").to_dict("records")

def effective_rates(profile, input_tokens):
    rates = (profile.get("input"), profile.get("output"))
    if None in rates:
        return None
    if input_tokens > profile.get("threshold", float("inf")):
        return (rates[0] * profile.get("input_multiplier", 1),
                rates[1] * profile.get("output_multiplier", 1))
    return rates

def api_cost(requests, input_tokens, output_tokens, rates):
    if rates is None:
        return None
    return requests * (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000

def render_comparison(timeline, directory):
    try:
        snapshot = json.loads((Path(directory) / "model_comparison_data.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        st.warning(f"Comparison details could not be loaded: {error}. Timeline choices and manual estimates remain available.")
        snapshot = {"profiles": {}, "checked": "Unavailable"}
    profiles = snapshot["profiles"]
    st.divider()
    st.header("Compare Models")
    st.caption("Compare two releases from the timeline. Some releases include several model versions; check the linked documentation for the version you plan to use.")
    st.html("""
    <style>
    .st-key-model-comparison {
        background: linear-gradient(135deg, #111c30, #0b1220);
        border: 1px solid #334155; border-radius: 16px;
        padding: 16px 18px 10px;
    }
    .st-key-model-comparison [data-testid="stMarkdownContainer"],
    .st-key-model-comparison [data-testid="stWidgetLabel"],
    .st-key-model-comparison [data-testid="stCaptionContainer"],
    .st-key-model-comparison [data-testid="stMetricLabel"],
    .st-key-model-comparison [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }
    .st-key-model-comparison h3 { color: #f1f5f9; }
    .st-key-model-comparison [data-testid="stWidgetLabel"] {
        justify-content: flex-start !important; gap: 6px !important;
    }
    .st-key-model-comparison [data-testid="stWidgetLabel"] > div {
        flex: 0 1 auto !important; width: auto !important;
    }
    .st-key-model-comparison [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] {
        margin-left: 0 !important; flex: 0 0 auto !important;
    }
    .st-key-model-comparison a { color: #93c5fd; }
    .st-key-model-comparison [data-testid="stTooltipIcon"] button { color: #93c5fd !important; }
    .st-key-model-comparison [data-testid="stTooltipIcon"] svg,
    .st-key-model-comparison [data-testid="stTooltipIcon"] svg * { stroke: #93c5fd !important; }
    .st-key-model-comparison [data-testid="stDownloadButton"] button {
        background: #1e293b; color: #f1f5f9; border: 1px solid #475569;
    }
    .st-key-model-comparison .comparison-factors {
        width: 100%; table-layout: fixed; border-collapse: collapse;
        color: #e2e8f0; font-size: 14px; line-height: 1.55;
    }
    .st-key-model-comparison .comparison-factors th,
    .st-key-model-comparison .comparison-factors td {
        padding: 12px; border: 1px solid #334155;
        text-align: left; vertical-align: top; overflow-wrap: anywhere;
    }
    .st-key-model-comparison .comparison-factors th { background: #1e293b; }
    .st-key-model-comparison .comparison-factors th:first-child { width: 22%; }
    .factor-help { display: inline; margin-left: 7px; position: relative; }
    .factor-help summary { display: inline; color: #93c5fd; cursor: pointer; list-style: none; }
    .factor-help summary::-webkit-details-marker { display: none; }
    .factor-help span { display: block; padding: 8px; margin-top: 6px;
        background: #26354a; border-radius: 6px; font-weight: normal; }
    </style>
    """)
    with st.container(key="model-comparison"):
        kind = st.radio("Comparison category", ["Closed", "Open-weight"], horizontal=True, key="comparison_kind", help="Choose models run by a provider or models with downloadable weights.")
        rows = model_options(timeline, kind)
        if len(rows) < 2:
            st.info("Add at least two releases in this category to timeline_data.csv.")
            return
        by_title = {row["title"]: row for row in rows}
        names = list(by_title)
        initial_a = next((i for i, name in enumerate(names) if kind == "Closed" and profiles.get(name, {}).get("input") is not None), 0)
        col_a, col_b = st.columns(2)
        with col_a:
            first = st.selectbox("Model A", names, index=initial_a, key=f"comparison_a_{kind}", help="Choose the first model you want to compare.")
        with col_b:
            other_names = [name for name in names if name != first]
            initial_b = next((i for i, name in enumerate(other_names) if kind == "Closed" and profiles.get(name, {}).get("input") is not None), 0)
            second = st.selectbox("Model B", other_names, index=initial_b,
                                  key=f"comparison_b_{kind}_{first}", help="Choose a different model from the same category to compare with Model A.")
        chosen = [by_title[first], by_title[second]]
        documentation_signature = json.dumps(
            [kind, [(row["title"], row["source"], profiles.get(row["title"], {}).get("documentation_urls"),
                     profiles.get(row["title"], {}).get("url")) for row in chosen]]
        )
        if st.button("Compare Models", type="primary", key="compare_models",
                     help="Read current official documentation and show the differences between your selected models."):
            clicked_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M:%S UTC")
            with st.spinner("Reading the selected models' documentation…"):
                live = compare_live_documents(chosen, profiles)
                live["clicked_at"] = clicked_at
                live["signature"] = documentation_signature
                st.session_state["comparison_documentation_result"] = live
        live = st.session_state.get("comparison_documentation_result")
        live_current = live is not None and live["signature"] == documentation_signature
        table = pd.DataFrame(columns=[first, second])
        table.index.name = "Factor"
        if live_current:
            st.subheader("Capabilities and decision factors")
            ordered = [factor for factor in FACTOR_HELP if factor in live["differing"]]
            if ordered:
                table = pd.DataFrame(
                    [live["differing"][factor] for factor in ordered],
                    index=ordered, columns=[first, second],
                )
                table.index.name = "Factor"
            elif all(model["documents"] for model in live["models"]):
                st.info("No documented differences were found in the factors that could be compared.")
            documentation = {
                row["title"]: source_urls(row, profiles.get(row["title"], {})) for row in chosen
            }
            st.html(comparison_html(table, documentation))
            for column, model in zip(st.columns(2), live["models"]):
                with column:
                    if not model["documents"]:
                        st.markdown("**" + model["title"] + "**")
                        reasons = [failure["reason"] for failure in model["failures"]]
                        blocked = any("10013" in reason or "access permissions" in reason for reason in reasons)
                        if blocked:
                            st.error("Documentation downloads are blocked by this app process's network permissions. Restart Streamlit from a terminal with internet access, then click Compare Models.")
                        else:
                            st.warning("Could not download this model's documentation. Check your internet connection and the Model Documentation links.")
                        with st.expander("Download error details"):
                            for failure in model["failures"]:
                                st.text(failure["url"] + "\n" + failure["reason"])
                    elif model["failures"]:
                        st.caption(model["title"] + ": some documentation could not be read; see Model Documentation.")
            st.caption("Documentation fetched " + live.get("clicked_at", live["fetched_at"]) + ". Factors without evidence for both models are omitted.")
            st.caption("These are documented capabilities and topics, not measured performance or a security rating.")
        else:
            st.caption("Select two models, then click Compare Models to read their current documentation.")
        st.subheader("Cost calculator")
        deployment = "Managed API"
        if kind == "Open-weight":
            deployment = st.radio("Cost scenario", ["Hosted API", "Self-hosted"], horizontal=True, key="comparison_deployment", help="Choose to pay a provider per use or run the model on infrastructure you pay for.")
        st.caption("Enter your expected usage, then click Compare Costs to calculate both models using these values.")
        request_count = st.number_input("Messages per month", min_value=1, value=10000, step=1000, key="comparison_requests", help="How many questions or messages you send to the model each month.")
        input_tokens = output_tokens = 0
        if deployment != "Self-hosted":
            st.caption("Text size is measured in tokens: small pieces of words. For English text, 100 words is roughly 130 tokens. Reply size should include any extra thinking tokens charged by the provider.")
            input_col, output_col = st.columns(2)
            with input_col:
                input_tokens = st.number_input("Average message size (tokens)", min_value=0, value=1000, step=100, key="comparison_input", help="The average size of everything sent with a question, including instructions and conversation history.")
            with output_col:
                output_tokens = st.number_input("Average reply size (tokens)", min_value=0, value=250, step=50, key="comparison_output", help="The average size of an answer, plus any extra thinking tokens the provider charges for.")
        costs = []
        calculation_inputs = []
        for column, row in zip(st.columns(2), chosen):
            with column:
                title = row["title"]
                key = f"comparison_{kind}_{deployment}_{title}"
                st.markdown(f"**{title}**")
                profile = profiles.get(title, {})
                if deployment == "Self-hosted":
                    infrastructure = st.number_input("Server and equipment cost ($ / month)", min_value=0.0, value=500.0, step=50.0, key=key + "_infra", help="Your monthly cost to rent or own the computers that run the model.")
                    operations = st.number_input("Maintenance and support cost ($ / month)", min_value=0.0, value=200.0, step=50.0, key=key + "_ops", help="Your monthly cost to keep the model running and get help when needed.")
                    cost = infrastructure + operations
                    calculation_inputs.append([infrastructure, operations])
                else:
                    defaults = effective_rates(profile, input_tokens) if kind == "Closed" else None
                    if defaults:
                        st.caption("Published prices are filled in below. You can edit them for this estimate.")
                        if profile.get("url"):
                            st.caption(f"[Price source]({profile['url']}) · Checked {snapshot['checked']}")
                    else:
                        st.caption("Enter both prices below to calculate this model's cost.")
                    rate_key = key + "_prices_" + str(defaults)
                    rate_in = st.number_input("Price for messages ($ / 1 million tokens)", min_value=0.0, value=float(defaults[0]) if defaults else None, step=0.1, format="%.4f", key=rate_key + "_in", help="The price for one million small pieces of text sent to the model.")
                    rate_out = st.number_input("Price for replies ($ / 1 million tokens)", min_value=0.0, value=float(defaults[1]) if defaults else None, step=0.1, format="%.4f", key=rate_key + "_out", help="The price for one million small pieces of text generated by the model, including billed thinking.")
                    rates = (rate_in, rate_out) if rate_in is not None and rate_out is not None else None
                    cost = api_cost(request_count, input_tokens, output_tokens, rates)
                    calculation_inputs.append(rates)
                costs.append(cost)
        signature = json.dumps([kind, first, second, deployment, request_count,
                                input_tokens, output_tokens, calculation_inputs])
        if st.button("Compare Costs", type="primary", key="compare_costs",
                     help="Calculate both models' costs from the values currently entered above."):
            if any(cost is None for cost in costs):
                missing = [row["title"] for row, cost in zip(chosen, costs) if cost is None]
                st.session_state.pop("comparison_cost_result", None)
                st.warning("Enter both message and reply prices for: " + ", ".join(missing))
            else:
                st.session_state["comparison_cost_result"] = {
                    "signature": signature, "costs": costs,
                }
        result = st.session_state.get("comparison_cost_result")
        current_result = result is not None and result["signature"] == signature
        if current_result:
            st.subheader("Cost comparison")
            for column, row, cost in zip(st.columns(2), chosen, result["costs"]):
                with column:
                    st.markdown(f'**{row["title"]}**')
                    st.metric("Estimated monthly cost", "$" + f"{cost:,.2f}",
                              help="The estimated total cost for this model for one month at your entered usage.")
                    st.metric("Cost per 1,000 messages", "$" + f"{cost / request_count * 1000:,.2f}",
                              help="The estimated average cost of sending 1,000 prompts at your entered usage.")
            st.metric("Monthly cost difference", "$" + f"{abs(costs[0] - costs[1]):,.2f}",
                      help="How much the two estimated monthly costs differ.")
        else:
            st.caption("Click Compare Costs to show results for your current values.")
        st.caption("Estimates in US dollars use your entered values and the displayed rates. API estimates cover one uncached call per request; extra tools, storage, retries, taxes and discounts are excluded. Self-hosted estimates add your infrastructure and support costs; check that the setup can handle your usage.")
        exported_table = table.copy()
        exported_table.loc["Estimated monthly cost (USD)"] = result["costs"] if current_result else [None, None]
        exported_table.loc["Calculation status"] = "Calculated" if current_result else "Click Compare Costs to calculate"
        exported_table.loc["Cost scenario"] = deployment
        exported_table.loc["Requests per month"] = request_count
        exported_table.loc["Billed input / output tokens per request"] = f"{input_tokens} / {output_tokens}"
        exported_table.loc["Documentation fetched"] = live["fetched_at"] if live_current else "Not fetched for this selection"
        exported_table.loc["Compare Models clicked"] = live.get("clicked_at", live["fetched_at"]) if live_current else "Not clicked for this selection"
        if live_current:
            for title in [first, second]:
                exported_table.loc["Model Documentation", title] = "; ".join(documentation[title])
                exported_table.loc["Documentation sources", title] = "; ".join(
                    next(model["documents"] for model in live["models"] if model["title"] == title)
                )
        exported_table.loc["Price snapshot checked"] = snapshot["checked"]
        export = exported_table.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download comparison (CSV)", export, "model_comparison.csv", "text/csv", key="comparison_download", help="Save the model details and any calculated costs as a spreadsheet-friendly CSV file.")

