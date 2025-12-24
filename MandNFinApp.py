import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="M&N Financial",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------
def compute_tax(income, deduction, brackets_df, credits=0.0):
    """
    Simple progressive tax calculator.
    brackets_df: DataFrame with columns ['upper_limit', 'rate'].
      - upper_limit in dollars (None or NaN means "no upper limit" = top bracket)
      - rate as decimal (e.g., 0.10 for 10%)
    Returns: tax_before_credits, tax_after_credits
    """
    taxable = max(0.0, income - deduction)
    remaining = taxable
    tax = 0.0
    last_limit = 0.0
    marginal_rate = 0.0

    # Sort by upper_limit (NaNs last)
    df = brackets_df.copy()
    df = df.sort_values(
        by="upper_limit",
        key=lambda col: col.fillna(float("inf"))
    )

    bracket_rows = []

    for _, row in df.iterrows():
        rate = float(row["rate"])
        upper = row["upper_limit"]
        if pd.isna(upper):
            # top bracket: everything remaining
            band_width = remaining
            band_tax = band_width * rate
            tax += band_tax
            marginal_rate = rate if band_width > 0 else marginal_rate
            bracket_rows.append(
                {
                    "upper_limit": None,
                    "rate": rate,
                    "income_in_bracket": band_width,
                    "tax_in_bracket": band_tax,
                }
            )
            remaining = 0.0
            break
        upper = float(upper)
        if remaining <= 0:
            bracket_rows.append(
                {
                    "upper_limit": upper,
                    "rate": rate,
                    "income_in_bracket": 0.0,
                    "tax_in_bracket": 0.0,
                }
            )
            last_limit = upper
            continue

        band_width = max(0.0, min(remaining, upper - last_limit))
        band_tax = band_width * rate
        tax += band_tax
        remaining -= band_width
        last_limit = upper
        if band_width > 0:
            marginal_rate = rate

        bracket_rows.append(
            {
                "upper_limit": upper,
                "rate": rate,
                "income_in_bracket": band_width,
                "tax_in_bracket": band_tax,
            }
        )


    tax_before = tax
    tax_after = max(0.0, tax_before - max(0.0, credits))
    effective_rate = tax_after / taxable if taxable > 0 else 0.0
    details_df = pd.DataFrame(bracket_rows)

    return taxable, tax_before, tax_after, effective_rate, marginal_rate, details_df

# Alternative Minimum Tax (AMT), which is basically a second tax system the IRS
# uses as a safety net so high-income taxpayers can’t reduce their regular tax too much with deductions.
def compute_simplified_amt(
    filing_status: str,
    taxable_income: float,
    amt_addbacks: float,
    amt_exemption: float,
    amt_phaseout_start: float,
    amt_rate: float,
) -> dict:
    """
    Very simplified AMT:
      AMTI = taxable_income + amt_addbacks
      exemption is reduced by 25% of (AMTI - phaseout_start), not below 0
      AMT taxable = max(0, AMTI - exemption)
      AMT tax = AMT taxable * amt_rate

    Returns dict with intermediate values.
    """
    amti = max(0.0, taxable_income + amt_addbacks)

    if amti <= amt_phaseout_start:
        exemption = amt_exemption
    else:
        reduction = 0.25 * (amti - amt_phaseout_start)
        exemption = max(0.0, amt_exemption - reduction)

    amt_taxable = max(0.0, amti - exemption)
    amt_tax = amt_taxable * amt_rate

    return {
        "AMTI": amti,
        "exemption": exemption,
        "amt_taxable": amt_taxable,
        "amt_tax": amt_tax,
    }



def format_currency(x):
    return f"${x:,.2f}"




# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("💻 Navigation")
page = st.sidebar.radio(
    "Select a tool:",
    ["Budget Planner", "Tax Estimator"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Simple personal finance + tax sandbox. Not tax advice 🙂")


# ==========================================================
# PAGE 1: BUDGET PLANNER
# ==========================================================
if page == "Budget Planner":
    st.title("💸 M&N Financial: Budget Planner")

    st.markdown(
        "Enter **monthly** income and expenses to see net cash flow, "
        "breakdowns, and exportable summaries."
    )
    colB1, colB2 = st.columns(2)
    with colB1:
        st.subheader("Income (Monthly)")
        with st.expander("Income sources", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                inc1_name = st.text_input("Income 1 label", "Primary job")
                inc1 = st.number_input(f"{inc1_name}", min_value=0.0, value=5000.0, step=100.0)
                inc2_name = st.text_input("Income 2 label", "Side job / freelance")
                inc2 = st.number_input(f"{inc2_name}", min_value=0.0, value=0.0, step=50.0)
            with col2:
                inc3_name = st.text_input("Income 3 label", "Partner / other")
                inc3 = st.number_input(f"{inc3_name}", min_value=0.0, value=0.0, step=50.0)
                inc4_name = st.text_input("Income 4 label", "Other income")
                inc4 = st.number_input(f"{inc4_name}", min_value=0.0, value=0.0, step=50.0)

        incomes = [
            (inc1_name, inc1),
            (inc2_name, inc2),
            (inc3_name, inc3),
            (inc4_name, inc4),
        ]

        total_income_monthly = sum(v for _, v in incomes) # inc1 + inc2 + inc3 + inc4
        total_income_annual = total_income_monthly * 12
    
    with colB2:
        st.subheader("Expenses (Monthly)")
        with st.expander("Common expenses", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                rent = st.number_input("Rent / Mortgage", min_value=0.0, value=1800.0, step=50.0)
                utilities = st.number_input("Utilities (electric, water, gas)", min_value=0.0, value=200.0, step=25.0)
                groceries = st.number_input("Groceries", min_value=0.0, value=400.0, step=25.0)
                transport = st.number_input("Transport (gas, transit)", min_value=0.0, value=150.0, step=25.0)
            with c2:
                car_payment = st.number_input("Car payment", min_value=0.0, value=0.0, step=25.0)
                insurance = st.number_input("Insurance (health/auto/etc.)", min_value=0.0, value=300.0, step=25.0)
                debt = st.number_input("Debt payments (loans/credit cards)", min_value=0.0, value=0.0, step=25.0)
                other_exp = st.number_input("Other expenses (entertainment, misc.)", min_value=0.0, value=300.0, step=25.0)

        expenses = {
            "Rent / Mortgage": rent,
            "Utilities": utilities,
            "Groceries": groceries,
            "Transport": transport,
            "Car payment": car_payment,
            "Insurance": insurance,
            "Debt payments": debt,
            "Other expenses": other_exp,
        }

        total_expenses_monthly = sum(expenses.values())
        total_expenses_annual = total_expenses_monthly * 12

        net_monthly = total_income_monthly - total_expenses_monthly
        net_annual = net_monthly * 12
        savings_rate = (net_monthly / total_income_monthly * 100.0) if total_income_monthly > 0 else 0.0


    st.markdown("### Summary")
    colA, colB, colC = st.columns(3)
    colA.metric("Monthly Income", format_currency(total_income_monthly))
    colB.metric("Monthly Expenses", format_currency(total_expenses_monthly))
    colC.metric(
        "Monthly Net",
        format_currency(net_monthly),
        delta=f"{savings_rate:0.1f}% of income"
    )

    colA2, colB2 = st.columns(2)
    colA2.metric("Annual Income", format_currency(total_income_annual))
    colB2.metric("Annual Net", format_currency(net_annual))

    # ---------------- VISUALIZATIONS ----------------
    st.markdown("---")
    st.markdown("### Visualizations")

    # 1) Income vs Expenses bar
    st.markdown("#### Income vs. Expenses (Monthly)")
    df_budget = pd.DataFrame(
        {
            "Category": ["Income", "Expenses"],
            "Amount": [total_income_monthly, total_expenses_monthly],
        }
    )

    fig = px.bar(
        df_budget,
        x="Category",
        y="Amount",
        text="Amount",  # show values on top of bars
    )

    # Optional: format text / axes a bit
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(
        yaxis_title="Amount ($)",
        xaxis_title="",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        bargap=.5,
    )

    st.plotly_chart(fig, use_container_width=True)


    # 2) Expense breakdown
    st.markdown("#### Expense Breakdown by Category (Monthly)")
    expenses_df = pd.DataFrame(
        {"Category": list(expenses.keys()), "Amount": list(expenses.values())}
    )
    #st.bar_chart(expenses_df.set_index("Category"))
    fig_exp = px.bar(
    expenses_df,
    x="Category",
    y="Amount",
    text="Amount",  # show values on bars
    )

    fig_exp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_exp.update_layout(
        yaxis_title="Amount ($)",
        xaxis_title="",
        xaxis_tickangle=-45,   # tilt labels if many categories
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        bargap=.5,
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    # 3) Income sources table
    st.markdown("#### Income Sources")
    income_df = pd.DataFrame(
        {"Source": [name for name, _ in incomes], "Monthly Amount": [val for _, val in incomes]}
    )
    st.dataframe(income_df.style.format({"Monthly Amount": "{:,.2f}"}), use_container_width=True)

# ---------------- PRINT / DOWNLOAD ----------------
    st.markdown("---")
    st.markdown("### Download / Print Summary")

    # Budget CSV
    budget_summary_df = pd.DataFrame(
        {
            "Metric": [
                "Monthly Income",
                "Monthly Expenses",
                "Monthly Net",
                "Annual Income",
                "Annual Expenses",
                "Annual Net",
                "Savings Rate (%)",
            ],
            "Value": [
                total_income_monthly,
                total_expenses_monthly,
                net_monthly,
                total_income_annual,
                total_expenses_annual,
                net_annual,
                savings_rate,
            ],
        }
    )
    budget_csv = budget_summary_df.to_csv(index=False)
    st.download_button(
        "Download budget summary as CSV",
        data=budget_csv,
        file_name="budget_summary.csv",
        mime="text/csv",
    )

    # Printable text summary
    budget_text_lines = [
        "BUDGET SUMMARY",
        "==============================",
        f"Monthly Income:  {format_currency(total_income_monthly)}",
        f"Monthly Expenses:{format_currency(total_expenses_monthly)}",
        f"Monthly Net:     {format_currency(net_monthly)}",
        "",
        f"Annual Income:   {format_currency(total_income_annual)}",
        f"Annual Expenses: {format_currency(total_expenses_annual)}",
        f"Annual Net:      {format_currency(net_annual)}",
        "",
        f"Savings Rate:    {savings_rate:0.1f}%",
        "",
        "Income sources:",
    ]
    for name, val in incomes:
        budget_text_lines.append(f"  - {name}: {format_currency(val)} / month")

    budget_text_lines.append("")
    budget_text_lines.append("Expense breakdown:")
    for k, v in expenses.items():
        budget_text_lines.append(f"  - {k}: {format_currency(v)} / month")

    budget_text = "\n".join(budget_text_lines)

    st.download_button(
        "Download printable budget report (TXT)",
        data=budget_text,
        file_name="budget_report.txt",
        mime="text/plain",
    )


    st.caption("Tip: after downloading the TXT, you can open it and print directly.")


# ==========================================================
# PAGE 2: TAX ESTIMATOR
# ==========================================================
else:
    st.title("🧾 M&N Financial: Tax Estimator")
    st.markdown(
        "This is a **toy** tax model so you can simulate **standard vs itemized** deductions, "
        "**brackets**, and **marginal rates**. It is **not** official tax advice."
    )

    st.subheader("Basic Inputs")

    col1, col2 = st.columns(2)
    with col1:
        filing_status = st.selectbox(
            "Filing status (label only)",
            ["Single", "Married filing jointly", "Married filing separately", "Head of household"],
            index=0,
        )
        gross_income = st.number_input(
            "Total taxable income before deductions (e.g. W-2 wages)",
            min_value=0.0,
            value=98_304.0,
            step=1_000.0,
        )
    with col2:
        tax_withheld = st.number_input(
            "Federal tax withheld (from W-2 etc.)",
            min_value=0.0,
            value=13_899.54,
            step=100.0,
        )
        credits = st.number_input(
            "Total tax credits (non-refundable) – optional",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )

    st.markdown("### Deductions")

    deduction_mode = st.radio(
        "Choose deduction type:",
        ["Standard deduction", "Itemized deduction"],
        index=0,
        horizontal=True,
    )

    # Map filing status → default standard deduction
    STD_DED_MAP = {
        "Single": 15_000.0,
        "Married filing jointly": 30_000.0,
        "Married filing separately": 15_000.0,
        "Head of household": 22_500.0,
    }

    if deduction_mode == "Standard deduction":
        # Initialize tracking of last filing status
        if "last_filing_status" not in st.session_state:
            st.session_state.last_filing_status = filing_status

        # If filing status changed, reset to the mapped default
        if filing_status != st.session_state.last_filing_status:
            st.session_state.standard_deduction = STD_DED_MAP.get(filing_status, 15_000.0)
            st.session_state.last_filing_status = filing_status

        # If not set yet, initialize from map
        if "standard_deduction" not in st.session_state:
            st.session_state.standard_deduction = STD_DED_MAP.get(filing_status, 15_000.0)

        standard_ded = st.number_input(
            "Standard deduction (auto from filing status, but editable)",
            min_value=0.0,
            value=st.session_state.standard_deduction,
            step=500.0,
        )
        st.session_state.standard_deduction = standard_ded
        deduction = standard_ded

        st.caption(
            f"Current default for **{filing_status}** is $"
            f"{STD_DED_MAP[filing_status]:,.0f}, but you can override it above."
        )
    else:
        st.info("Enter itemized deductions. Leave unused fields as 0.")
        c1, c2 = st.columns(2)
        with c1:
            item_mortgage = st.number_input("Mortgage interest", min_value=0.0, value=0.0, step=100.0)
            item_state_tax = st.number_input("State & local taxes", min_value=0.0, value=0.0, step=100.0)
            item_charity = st.number_input("Charitable contributions", min_value=0.0, value=0.0, step=100.0)
        with c2:
            item_medical = st.number_input("Medical (allowed portion)", min_value=0.0, value=0.0, step=100.0)
            item_other = st.number_input("Other itemized deductions", min_value=0.0, value=0.0, step=100.0)
        deduction = item_mortgage + item_state_tax + item_charity + item_medical + item_other

        st.markdown(f"**Total itemized deduction**: {format_currency(deduction)}")


    st.markdown("---")
    st.markdown("### Tax Brackets / Marginal Rates (Editable)")

    st.caption(
        "Edit the upper limits and rates to match IRS tax year adjustments. "
        "Leave the last upper limit empty (None) to indicate the top bracket."
    )

    # ---- Default brackets by filing status ----
    # Using the thresholds you provided
    BRACKET_LIMITS = {
        "Single": [11_925.0, 48_475.0, 103_350.0, 197_300.0, 250_525.0, 626_350.0, None],
        "Married filing jointly": [23_850.0, 96_950.0, 206_700.0, 394_600.0, 501_050.0, 751_600.0, None],
        "Head of household": [17_000.0, 64_850.0, 103_350.0, 197_300.0, 250_500.0, 626_350.0, None],
        "Married filing separately": [11_925.0, 48_475.0, 103_350.0, 197_300.0, 250_525.0, 375_800.0, None],
    }

    BRACKET_RATES = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]


    def default_brackets_df(status: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "upper_limit": BRACKET_LIMITS.get(status, BRACKET_LIMITS["Single"]),
                "rate": BRACKET_RATES,
            }
        )


    # ---- Initialize / update session_state when filing status changes ----
    if "last_bracket_status" not in st.session_state:
        st.session_state.last_bracket_status = filing_status

    if "brackets_df" not in st.session_state:
        # First time: use defaults for current filing status
        st.session_state.brackets_df = default_brackets_df(filing_status)
    else:
        # If user changed filing status, reset to that status's defaults
        if filing_status != st.session_state.last_bracket_status:
            st.session_state.brackets_df = default_brackets_df(filing_status)
            st.session_state.last_bracket_status = filing_status

    # Optional: let user reset brackets manually
    reset_brackets = st.button("Reset brackets to defaults for this filing status")
    if reset_brackets:
        st.session_state.brackets_df = default_brackets_df(filing_status)

    # ---- Editable table ----
    brackets_df = st.data_editor(
        st.session_state.brackets_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "upper_limit": st.column_config.NumberColumn(
                "Upper limit ($)",
                help=(
                    "Taxable income up to this amount is taxed at the given rate. "
                    "Leave blank in the last row to indicate 'no upper limit'."
                ),
                min_value=0.0,
                step=1_000.0,
            ),
            "rate": st.column_config.NumberColumn(
                "Rate",
                help="Marginal rate as a decimal. For 10% enter 0.10.",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
        },
    )
    st.session_state.brackets_df = brackets_df


    # Compute regular tax + details
    taxable, tax_before, tax_after, eff_rate, marg_rate, detail_df = compute_tax(
        gross_income, deduction, brackets_df, credits=credits
    )
    refund_or_due = tax_withheld - tax_after
    net_after_tax = max(0.0, gross_income - tax_after)



    st.markdown("---")
    st.markdown("### Alternative Minimum Tax (AMT) (Optional)")

    use_amt = st.checkbox(
        "Enable simplified AMT calculation",
        value=False,
        help=(
            "This uses a simplified AMT model: AMTI = taxable_income + add-backs, "
            "minus an exemption (with phaseout), taxed at a single AMT rate. "
            "Final tax is max(regular tax, AMT)."
        ),
    )

    # ---- Default AMT exemption & phaseout based on filing status ----
    if filing_status == "Married filing jointly":
        default_exemption = 137_000.0
        default_phaseout = 1_252_700.0
    elif filing_status == "Married filing separately":
        default_exemption = 68_650.0
        default_phaseout = 626_350.0
    elif filing_status == "Head of household":
        default_exemption = 88_100.0
        default_phaseout = 626_350.0
    else:  # "Single" / generic unmarried
        default_exemption = 88_100.0
        default_phaseout = 626_350.0

    amt_col1, amt_col2, amt_col3 = st.columns(3)

    with amt_col1:
        amt_exemption = st.number_input(
            "AMT exemption ($)",
            min_value=0.0,
            value=default_exemption,
            step=1_000.0,
            help="Base AMT exemption before phaseout.",
        )
    with amt_col2:
        amt_phaseout_start = st.number_input(
            "AMT phaseout starts at ($)",
            min_value=0.0,
            value=default_phaseout,
            step=10_000.0,
            help="AMTI level at which the AMT exemption starts to phase out.",
        )
    with amt_col3:
        amt_rate = st.number_input(
            "AMT rate",
            min_value=0.0,
            max_value=1.0,
            value=0.26,
            step=0.01,
            help="Single blended AMT rate (e.g. 0.26 for 26%).",
        )

    amt_addbacks = st.number_input(
        "AMT add-backs / adjustments ($)",
        min_value=0.0,
        value=0.0,
        step=1_000.0,
        help="Approximate total of AMT adjustments and preference items added to taxable income.",
    )

    if use_amt:
        amt_info = compute_simplified_amt(
            filing_status=filing_status,
            taxable_income=taxable,
            amt_addbacks=amt_addbacks,
            amt_exemption=amt_exemption,
            amt_phaseout_start=amt_phaseout_start,
            amt_rate=amt_rate,
        )
        amt_tax = amt_info["amt_tax"]
    else:
        amt_info = None
        amt_tax = 0.0

    # Add AMT to after tax
    # --- Regular tax after credits (already computed) ---
    regular_tax = tax_after  # rename for clarity

    # --- Final tax after AMT (if enabled) ---
    if use_amt:
        final_tax = max(regular_tax, amt_tax)
    else:
        final_tax = regular_tax

    refund_or_due = tax_withheld - final_tax
    net_after_tax = max(0.0, gross_income - final_tax)


    st.markdown("---")
    st.markdown("### Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Taxable income", format_currency(taxable))
    c2.metric("Tax before credits", format_currency(tax_before))
    if use_amt:
        c3.metric("AMT (simplified)", format_currency(amt_tax))
    else:
        c3.metric("Tax after credits", format_currency(tax_after))

    c4, c5 = st.columns(2)
 
    c4.metric("Effective tax rate", f"{(final_tax / taxable * 100) if taxable > 0 else 0:0.2f}%")
    c5.metric("Marginal tax rate", f"{marg_rate*100:0.2f}%")


    if refund_or_due >= 0:
        st.success(f"Estimated **refund**: {format_currency(refund_or_due)}")
    else:
        st.error(f"Estimated **amount owed**: {format_currency(-refund_or_due)}")

    st.markdown("#### Quick summary")
    st.write(
        f"- Filing status (label): **{filing_status}**\n"
        f"- Gross income: **{format_currency(gross_income)}**\n"
        f"- Deduction used: **{format_currency(deduction)}** "
        f"({'Standard' if deduction_mode=='Standard deduction' else 'Itemized'})\n"
        f"- Tax withheld: **{format_currency(tax_withheld)}**\n"
        f"- Credits: **{format_currency(credits)}**\n"
        f"- Regular tax after credits: **{format_currency(regular_tax)}**\n"
        f"- AMT (simplified): **{format_currency(amt_tax if use_amt else 0.0)}**\n"
        f"- **Final tax (max of regular & AMT): {format_currency(final_tax)}**\n"
        f"- Net after tax: **{format_currency(net_after_tax)}**"
    )


    # ---------------- VISUALIZATIONS ----------------
    st.markdown("---")
    st.markdown("### Visualizations")

    # 1) Tax vs Net Income bar
    st.markdown("#### Tax vs. Net Income")
    tax_net_df = pd.DataFrame(
        {
            "Category": ["Gross income", "Tax (after credits)", "Net after tax"],
            "Amount": [gross_income, tax_after, net_after_tax],
        }
    )

    fig_exp = px.bar(
    tax_net_df,
    x="Category",
    y="Amount",
    text="Amount",  # show values on bars
    )

    fig_exp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_exp.update_layout(
        yaxis_title="Amount ($)",
        xaxis_title="",
        xaxis_tickangle=0,   # tilt labels if many categories
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        bargap=.5,
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    # 2) Per-bracket tax contribution
    st.markdown("#### Tax per Bracket")
    # Create a nicer label for brackets
    labels = []
    prev = 0.0
    for _, row in detail_df.iterrows():
        upper = row["upper_limit"]
        rate = row["rate"]
        if pd.isna(upper):
            lbl = f">{prev:,.0f} @ {rate*100:.1f}%"
        else:
            lbl = f"{prev:,.0f}–{upper:,.0f} @ {rate*100:.1f}%"
        labels.append(lbl)
        prev = upper if not pd.isna(upper) else prev

    detail_df["Bracket"] = labels

    st.dataframe(
        detail_df[["Bracket", "income_in_bracket", "tax_in_bracket"]]
        .rename(
            columns={
                "income_in_bracket": "Income in bracket",
                "tax_in_bracket": "Tax in bracket",
            }
        )
        .style.format(
            {
                "Income in bracket": "{:,.2f}",
                "Tax in bracket": "{:,.2f}",
            }
        ),
        use_container_width=True,
    )

    # ---------------- PRINT / DOWNLOAD ----------------
    st.markdown("---")
    st.markdown("### Download / Print Summary")

    # 1) Tax summary text
    tax_text_lines = [
        "TAX SUMMARY",
        "==============================",
        f"Filing status:      {filing_status}",
        "",
        f"Gross income:       {format_currency(gross_income)}",
        f"Deduction used:     {format_currency(deduction)} "
        f"({'Standard' if deduction_mode=='Standard deduction' else 'Itemized'})",
        f"Taxable income:     {format_currency(taxable)}",
        "",
        f"Tax before credits: {format_currency(tax_before)}",
        f"Credits:            {format_currency(credits)}",
        f"Tax after credits:  {format_currency(tax_after)}",
        "",
        f"Tax withheld:       {format_currency(tax_withheld)}",
        f"Net after tax:      {format_currency(net_after_tax)}",
        "",
        f"Effective tax rate: {eff_rate*100:0.2f}%",
        f"Marginal tax rate:  {marg_rate*100:0.2f}%",
        "",
        "Bracket breakdown:",
    ]
    for _, row in detail_df.iterrows():
        tax_text_lines.append(
            f"  - {row['Bracket']}: "
            f"income {format_currency(row['income_in_bracket'])}, "
            f"tax {format_currency(row['tax_in_bracket'])}"
        )

    tax_text = "\n".join(tax_text_lines)
    colP1, colP2 = st.columns(2)
    with colP1:
        st.download_button(
            "Download printable tax report (TXT)",
            data=tax_text,
            file_name="tax_report.txt",
            mime="text/plain",
        )
    with colP2:
        # 2) Bracket CSV
        bracket_csv = detail_df.to_csv(index=False)
        st.download_button(
            "Download bracket details as CSV",
            data=bracket_csv,
            file_name="tax_brackets_detail.csv",
            mime="text/csv",
        )

    st.caption(
        "You can open the TXT/CSV files in your editor or spreadsheet and print them if needed."
    )

    st.markdown("---")
    st.caption(
        "This is a simplified calculator for experimentation only. "
        "Real tax returns include many more rules, phase-outs, and special cases."
    )
