# Part B window (ALL previous months up to the day before A_start)
today = _pd.Timestamp.today().normalize()
# Part B spans from the earliest available Create Date to just before Part A starts
B_start = _pd.to_datetime(df[_create], errors='coerce').min()
B_end = A_start - _pd.Timedelta(days=1)

        def _period_table(start, end, label):
            d = df.copy()
            # enrolments: pay within period (fallback to created if no pay)
            if _pay: en = d[(d[_pay]>=start) & (d[_pay]<=end)].copy()
            else:    en = d[(d[_create]>=start) & (d[_create]<=end)].copy()
            de = d[(d[_create]>=start) & (d[_create]<=end)].copy()

            g_en = en.groupby([_src, cty_col]).size().rename("Enrollments")
            g_de = de.groupby([_src, cty_col]).size().rename("Deals")
            g = _pd.concat([g_en, g_de], axis=1).fillna(0).reset_index()
            g["Conversion %"] = _np.where(g["Deals"]>0, 100.0 * g["Enrollments"]/g["Deals"], _np.nan)

            # Calibration counts
            def _safe_count(col):
                if not col or col not in d.columns: return _pd.Series(dtype="float64")
                m = (d[col]>=start) & (d[col]<=end)
                return d[m].groupby([_src, cty_col]).size()
            cal1 = _safe_count(_cal1).rename("First Calibration Scheduled")
            calr = _safe_count(_calr).rename("Calibration Rescheduled")
            if _cald and _cald in d.columns:
                cald = _safe_count(_cald).rename("Calibration Done")
            else:
                if _stage and _stage in d.columns:
                    m_done = d[_stage].astype(str).str.contains("Calibration Done|Trial Done", case=False, na=False)
                    cald = d[m_done & (d[_create]>=start) & (d[_create]<=end)].groupby([_src, cty_col]).size().rename("Calibration Done")
                else:
                    cald = _pd.Series(dtype="float64")
            g = g.merge(cal1, on=[_src, cty_col], how="left")
            g = g.merge(calr, on=[_src, cty_col], how="left")
            g = g.merge(cald, on=[_src, cty_col], how="left")
            for c in ["First Calibration Scheduled","Calibration Rescheduled","Calibration Done"]:
                if c in g.columns: g[c] = g[c].fillna(0).astype(int)

            # Top-N selector
            st.markdown(f"**{label} — Country Selection**")
            all_toggle = st.checkbox(f"{label}: All countries", value=True, key=f"{label}_all")
            if not all_toggle:
                topn = st.number_input(f"{label}: Top N countries", min_value=1, max_value=50, value=3, step=1, key=f"{label}_topn")
                rank = g.groupby([_src, cty_col])["Enrollments"].sum().reset_index()
                keep_pairs = []
                for s in sel_srcs or g[_src].unique().tolist():
                    sub = rank[rank[_src]==s].sort_values("Enrollments", ascending=False)
                    keep_pairs += [(s, n) for n in sub[cty_col].head(topn).tolist()]
                mask = g.apply(lambda r: (r[_src], r[cty_col]) in set(keep_pairs), axis=1)
                g = g[mask].copy()
            names = sorted(g[cty_col].unique().tolist())
            st.caption(f"{label} Countries: " + (", ".join(names) if names else "—"))

            g = g.sort_values([_src, "Enrollments"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(g, use_container_width=True)
            st.download_button(f"Download {label} CSV", data=g.to_csv(index=False), file_name=f"marketing_plan_{label.replace(' ','_').lower()}.csv", mime="text/csv")
            return g

        GA = _period_table(A_start, A_end, "Part A")
        st.markdown("### Part B (Previous Period)")
        GB = _period_table(B_start, B_end, "Part B")

    _marketing_plan_tab()

# --- Kids detail dispatch ---
if view == "Kids detail":
    _render_marketing_kids_detail(df)
# --- /Kids detail dispatch ---
if view == "Master Graph":
    def _master_graph_tab():
        import pandas as pd, numpy as np, altair as alt
        from datetime import date
        st.subheader("Master Graph — Flexible Visuals (MTD / Cohort)")

        # ---------- Resolve columns (defensive)
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, [
            "Create Date","Created Date","Deal Create Date","CreateDate","Created On"
        ])
        _pay    = pay_col    if (pay_col    in df_f.columns) else find_col(df_f, [
            "Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"
        ])
        _first  = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, [
            "First Calibration Scheduled Date","First Calibration","First Cal Scheduled"
        ])
        _resch  = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, [
            "Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"
        ])
        _done   = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, [
            "Calibration Done Date","Cal Done Date","Calibration Completed"
        ])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Master Graph needs 'Create Date' and 'Payment Received Date'. Please map them in the sidebar.", icon="⚠️")
            st.stop()

        # ---------- Controls: mode, scope, granularity
        c0, c1, c2, c3 = st.columns([1.0, 1.0, 1.1, 1.1])
        with c0:
            mode = st.radio("Mode", ["MTD","Cohort"], index=0, horizontal=True, key="mg_mode",
                            help=("MTD: count events only when the deal was also created in the same period;"
                                  " Cohort: count events by their own date (create can be anywhere)."))
        with c1:
            gran = st.radio("Granularity", ["Day","Week","Month"], index=2, horizontal=True, key="mg_gran")
        today_d = date.today()
        with c2:
            c_start = st.date_input("Create start", value=today_d.replace(day=1), key="mg_cstart")
        with c3:
            c_end   = st.date_input("Create end",   value=month_bounds(today_d)[1], key="mg_cend")
        if c_end < c_start:
            st.error("End date cannot be before start date.")
            st.stop()

        # ---------- Choose build type & chart type
        c4, c5 = st.columns([1.2, 1.2])
        with c4:
            build_type = st.radio("Build", ["Single metric","Combined (dual-axis)","Derived ratio"], index=0, horizontal=True, key="mg_build")
        with c5:
            chart_type = st.selectbox(
                "Chart type",
                ["Line","Bar","Area","Stacked Bar","Histogram","Bell Curve"],
                index=0,
                key="mg_chart",
                help="Histogram/Bell Curve apply to daily/period counts of a single metric."
            )

        # ---------- Normalize event timestamps
        C = coerce_datetime(df_f[_create])
        P = coerce_datetime(df_f[_pay])
        F = coerce_datetime(df_f[_first]) if (_first and _first in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
        R = coerce_datetime(df_f[_resch]) if (_resch and _resch in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
        D = coerce_datetime(df_f[_done])  if (_done  and _done  in df_f.columns)  else pd.Series(pd.NaT, index=df_f.index)

        # Period keys by granularity
        def _per(s):
            ds = pd.to_datetime(s, errors="coerce")
            if gran == "Day":
                return ds.dt.floor("D")
            if gran == "Week":
                # ISO week start Monday
                return (ds - pd.to_timedelta(ds.dt.weekday, unit="D")).dt.floor("D")
            return ds.dt.to_period("M").dt.to_timestamp()

        perC, perP, perF, perR, perD = _per(C), _per(P), _per(F), _per(R), _per(D)

        # Universe = created within [c_start, c_end]
        C_date = C.dt.date
        in_window = C_date.notna() & (C_date >= c_start) & (C_date <= c_end)

        # MTD requires event period == create period; Cohort uses event’s own period
        sameP = perC == perP
        sameF = perC == perF
        sameR = perC == perR
        sameD = perC == perD

        if mode == "MTD":
            m_created = in_window & C.notna()
            m_enrol   = in_window & P.notna() & sameP
            m_first   = in_window & F.notna() & sameF
            m_resch   = in_window & R.notna() & sameR
            m_done    = in_window & D.notna() & sameD
        else:
            m_created = in_window & C.notna()
            m_enrol   = in_window & P.notna()
            m_first   = in_window & F.notna()
            m_resch   = in_window & R.notna()
            m_done    = in_window & D.notna()

        # Metric → (period_series, mask)
        metric_defs = {
            "Deals Created":              (perC, m_created),
            "Enrolments":                 (perP, m_enrol),
            "First Cal Scheduled":        (perF, m_first),
            "Cal Rescheduled":            (perR, m_resch),
            "Cal Done":                   (perD, m_done),
        }
        metric_names = list(metric_defs.keys())

        # ---------- Metric pickers (depend on build type)
        if build_type == "Single metric":
            m1 = st.selectbox("Metric", metric_names, index=0, key="mg_m1")
        elif build_type == "Combined (dual-axis)":
            cA, cB = st.columns(2)
            with cA:
                m1 = st.selectbox("Left Y", metric_names, index=0, key="mg_m1l")
            with cB:
                m2 = st.selectbox("Right Y", [m for m in metric_names if m != m1], index=0, key="mg_m2r")
        else:  # Derived ratio
            cA, cB = st.columns(2)
            with cA:
                num_m = st.selectbox("Numerator", metric_names, index=1, key="mg_num")
            with cB:
                den_m = st.selectbox("Denominator", [m for m in metric_names if m != num_m], index=0, key="mg_den")
            as_pct = st.checkbox("Show as % (×100)", value=True, key="mg_ratio_pct")

        # ---------- Helpers to aggregate counts by period
        def _count_series(per_s, mask, label):
            if mask is None or not mask.any():
                return pd.DataFrame(columns=["Period", label])
            df = pd.DataFrame({"Period": per_s[mask]})
            if df.empty:
                return pd.DataFrame(columns=["Period", label])
            return df.assign(_one=1).groupby("Period")["_one"].sum().rename(label).reset_index()

        # ---------- Build outputs
        if build_type == "Single metric":
            per_s, msk = metric_defs[m1]
            counts = _count_series(per_s, msk, m1)

            # Graph / Histogram / Bell
            view = st.radio("View as", ["Graph","Table"], index=0, horizontal=True, key="mg_view_single")

            if view == "Table":
                st.dataframe(counts.sort_values("Period"), use_container_width=True)
                st.download_button("Download CSV", counts.sort_values("Period").to_csv(index=False).encode("utf-8"),
                                   "master_graph_single.csv","text/csv", key="mg_dl_single")
            else:
                if chart_type in {"Histogram","Bell Curve"}:
                    # Build distribution of period counts (e.g., daily counts)
                    vals = counts[m1].astype(float)
                    if vals.empty:
                        st.info("No data to plot a distribution.")
                    else:
                        hist = alt.Chart(counts).mark_bar(opacity=0.9).encode(
                            x=alt.X(f"{m1}:Q", bin=alt.Bin(maxbins=30), title=f"{m1} per {gran}"),
                            y=alt.Y("count():Q", title="Frequency"),
                            tooltip=[alt.Tooltip("count():Q", title="Freq")]
                        ).properties(height=320, title=f"Histogram — {m1} per {gran}")

                        if chart_type == "Bell Curve":
                            mu  = float(vals.mean())
                            sig = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
                            # synthetic normal
                            xs = np.linspace(max(0, vals.min()), vals.max() if vals.max()>0 else 1.0, 200)
                            # scale PDF to same area ~ total count of bars
                            pdf = (1.0/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs-mu)/max(sig,1e-9))**2)) if sig>0 else np.zeros_like(xs)
                            pdf = pdf / pdf.max() * (counts["count()"].max() if "count()" in counts.columns else 1.0) if sig>0 else pdf
                            bell_df = pd.DataFrame({"x": xs, "pdf": pdf})
                            bell = alt.Chart(bell_df).mark_line().encode(
                                x=alt.X("x:Q", title=f"{m1} per {gran}"),
                                y=alt.Y("pdf:Q", title="Density (scaled)")
                            )
                            st.altair_chart(hist + bell, use_container_width=True)
                            st.caption(f"μ = {mu:.2f}, σ = {sig:.2f}")
                        else:
                            st.altair_chart(hist, use_container_width=True)
                else:
                    mark = {"Line":"line","Bar":"bar","Area":"area","Stacked Bar":"bar"}.get(chart_type, "line")
                    base = alt.Chart(counts)
                    ch = (
                        base.mark_line(point=True) if mark=="line" else
                        base.mark_area(opacity=0.5) if mark=="area" else
                        base.mark_bar(opacity=0.9)
                    ).encode(
                        x=alt.X("Period:T", title="Period"),
                        y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    ).properties(height=360, title=f"{m1} by {gran}")
                    st.altair_chart(ch, use_container_width=True)

        elif build_type == "Combined (dual-axis)":
            per1, m1_mask = metric_defs[m1]
            per2, m2_mask = metric_defs[m2]
            s1 = _count_series(per1, m1_mask, m1)
            s2 = _count_series(per2, m2_mask, m2)
            combined = s1.merge(s2, on="Period", how="outer").fillna(0)

            view = st.radio("View as", ["Graph","Table"], index=0, horizontal=True, key="mg_view_combo")
            if view == "Table":
                st.dataframe(combined.sort_values("Period"), use_container_width=True)
                st.download_button("Download CSV", combined.sort_values("Period").to_csv(index=False).encode("utf-8"),
                                   "master_graph_combined.csv","text/csv", key="mg_dl_combined")
            else:
                # Dual-axis with layering (left = bars/line, right = line)
                if chart_type == "Bar":
                    left = alt.Chart(combined).mark_bar(opacity=0.85).encode(
                        x=alt.X("Period:T", title="Period"),
                        y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    )
                elif chart_type == "Area":
                    left = alt.Chart(combined).mark_area(opacity=0.5).encode(
                        x=alt.X("Period:T"), y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    )
                else:
                    left = alt.Chart(combined).mark_line(point=True).encode(
                        x=alt.X("Period:T"), y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    )

                right = alt.Chart(combined).mark_line(point=True).encode(
                    x=alt.X("Period:T"),
                    y=alt.Y(f"{m2}:Q", title=m2, axis=alt.Axis(orient="right")),
                    tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                             alt.Tooltip(f"{m2}:Q", format="d")]
                )

                st.altair_chart(alt.layer(left, right).resolve_scale(y='independent').properties(height=360,
                                  title=f"{m1} (left) + {m2} (right) by {gran}"), use_container_width=True)

        else:
            # Derived ratio
            perN, maskN = metric_defs[num_m]
            perD, maskD = metric_defs[den_m]
            sN = _count_series(perN, maskN, "Num")
            sD = _count_series(perD, maskD, "Den")
            ratio = sN.merge(sD, on="Period", how="outer").fillna(0.0)
            ratio["Value"] = np.where(ratio["Den"]>0, ratio["Num"]/ratio["Den"], np.nan)
            if as_pct:
                ratio["Value"] = ratio["Value"] * 100.0

            label = f"{num_m} / {den_m}" + (" (%)" if as_pct else "")
            ratio = ratio.rename(columns={"Value": label})

            view = st.radio("View as", ["Graph","Table"], index=0, horizontal=True, key="mg_view_ratio")
            if view == "Table":
                st.dataframe(ratio[["Period", label]].sort_values("Period"), use_container_width=True)
                st.download_button("Download CSV", ratio[["Period", label]].sort_values("Period").to_csv(index=False).encode("utf-8"),
                                   "master_graph_ratio.csv","text/csv", key="mg_dl_ratio")
            else:
                if chart_type in {"Histogram","Bell Curve"}:
                    vals = ratio[label].dropna().astype(float)
                    if vals.empty:
                        st.info("No data to plot a distribution.")
                    else:
                        hist = alt.Chart(ratio.dropna()).mark_bar(opacity=0.9).encode(
                            x=alt.X(f"{label}:Q", bin=alt.Bin(maxbins=30), title=label),
                            y=alt.Y("count():Q", title="Frequency"),
                            tooltip=[alt.Tooltip("count():Q", title="Freq")]
                        ).properties(height=320, title=f"Histogram — {label}")
                        if chart_type == "Bell Curve":
                            mu  = float(vals.mean())
                            sig = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
                            xs = np.linspace(vals.min(), vals.max() if vals.max()!=vals.min() else vals.min()+1.0, 200)
                            pdf = (1.0/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs-mu)/max(sig,1e-9))**2)) if sig>0 else np.zeros_like(xs)
                            pdf = pdf / pdf.max() *  (ratio["count()"].max() if "count()" in ratio.columns else 1.0)
                            bell_df = pd.DataFrame({"x": xs, "pdf": pdf})
                            bell = alt.Chart(bell_df).mark_line().encode(x="x:Q", y="pdf:Q")
                            st.altair_chart(hist + bell, use_container_width=True)
                            st.caption(f"μ = {mu:.3f}, σ = {sig:.3f}")
                        else:
                            st.altair_chart(hist, use_container_width=True)
                else:
                    mark = {"Line":"line","Bar":"bar","Area":"area","Stacked Bar":"bar"}.get(chart_type, "line")
                    base = alt.Chart(ratio)
                    ch = (
                        base.mark_line(point=True) if mark=="line" else
                        base.mark_area(opacity=0.5) if mark=="area" else
                        base.mark_bar(opacity=0.9)
                    ).encode(
                        x=alt.X("Period:T", title="Period"),
                        y=alt.Y(f"{label}:Q", title=label),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{label}:Q", format=".2f" if as_pct else ".3f")]
                    ).properties(height=360, title=f"{label} by {gran}")
                    st.altair_chart(ch, use_container_width=True)

    # run it
    _master_graph_tab()




# =========================================
# HubSpot Deal Score tracker (fresh build)
# =========================================
if view == "HubSpot Deal Score tracker":
    import pandas as pd, numpy as np
    import altair as alt
    from datetime import date, timedelta
    from calendar import monthrange

    st.subheader("HubSpot Deal Score tracker — Score Calibration & Month Prediction")

    # ---------- Helpers ----------
    def _pick(df, preferred, cands):
        if preferred and preferred in df.columns: return preferred
        for c in cands:
            if c in df.columns: return c
        return None

    def month_bounds(d: date):
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

    # ---------- Resolve columns ----------
    _create = _pick(df_f, globals().get("create_col"),
                    ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
    _pay    = _pick(df_f, globals().get("pay_col"),
                    ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
    _score  = _pick(df_f, None,
                    ["HubSpot Deal Score","HubSpot DLSCore","HubSpot DLS Score","Deal Score","HubSpot Score","DLSCore"])

    if not _create or not _pay or not _score:
        st.warning("Need columns: Create Date, Payment Received Date, and HubSpot Deal Score. Please map them.", icon="⚠️")
        st.stop()

    dfm = df_f.copy()
    dfm["__C"] = pd.to_datetime(dfm[_create], errors="coerce", dayfirst=True)
    dfm["__P"] = pd.to_datetime(dfm[_pay],    errors="coerce", dayfirst=True)
    dfm["__S"] = pd.to_numeric(dfm[_score], errors="coerce")  # score as float

    has_score = dfm["__S"].notna()

    # ---------- Controls ----------
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
    with c1:
        lookback = st.selectbox("Lookback (months, exclude current)", [3, 6, 9, 12], index=1)
    with c2:
        n_bins = st.selectbox("# Score bins", [6, 8, 10, 12, 15], index=2)
    with c3:
        ref_age_days = st.number_input("Normalization age (days)", min_value=7, max_value=120, value=30, step=1,
                                       help="Young deals have lower scores; normalize each score up to this age (cap at 1×).")

    # Current month scope
    today_d = date.today()
    mstart_cur, mend_cur = month_bounds(today_d)
    c4, c5 = st.columns(2)
    with c4:
        cur_start = st.date_input("Prediction window start", value=mstart_cur, key="hsdls_cur_start")
    with c5:
        cur_end   = st.date_input("Prediction window end",   value=mend_cur,   key="hsdls_cur_end")
    if cur_end < cur_start:
        st.error("Prediction window end cannot be before start.")
        st.stop()

    # ---------- Build Training (historical) ----------
    cur_per = pd.Period(today_d, freq="M")
    hist_months = [cur_per - i for i in range(1, lookback+1)]
    if not hist_months:
        st.info("No historical months selected.")
        st.stop()

    # A deal is in a historical month by its Create month
    dfm["__Cper"] = dfm["__C"].dt.to_period("M")
    hist_mask = dfm["__Cper"].isin(hist_months) & has_score
    # Label = converted (ever got a payment date)
    dfm["__converted"] = dfm["__P"].notna()

    hist_df = dfm.loc[hist_mask, ["__C","__P","__S","__converted"]].copy()

    if hist_df.empty:
        st.info("No historical rows with HubSpot Deal Score found in the selected lookback.")
        st.stop()

    # ---------- Normalization for "young" deals ----------
    # adjusted_score = score * min(ref_age_days / max(age_days,1), 1.0)
    age_days_hist = (pd.Timestamp(today_d) - hist_df["__C"]).dt.days.clip(lower=1)
    hist_df["__S_adj"] = hist_df["__S"] * np.minimum(ref_age_days / age_days_hist, 1.0)

    # ---------- Learn probability by score range ----------
    # Quantile-based bins for even coverage; fallback to linear if ties dominate
    try:
        q = np.linspace(0, 1, n_bins+1)
        edges = np.unique(np.nanquantile(hist_df["__S_adj"], q))
        if len(edges) < 3:
            raise ValueError
    except Exception:
        smin, smax = float(hist_df["__S_adj"].min()), float(hist_df["__S_adj"].max())
        if smax <= smin:
            smax = smin + 1e-6
        edges = np.linspace(smin, smax, n_bins+1)

    hist_df["__bin"] = pd.cut(hist_df["__S_adj"], bins=edges, include_lowest=True, right=True)

    # Laplace smoothing to avoid 0/100%
    grp = (hist_df.groupby("__bin", observed=True)["__converted"]
                 .agg(Total="count", Conversions="sum"))
    grp["Prob%"] = (grp["Conversions"] + 1) / (grp["Total"] + 2) * 100.0
    grp = grp.reset_index()
    grp["Range"] = grp["__bin"].astype(str)

    # ---------- Show Calibration (bins) ----------
    st.markdown("### Calibration: HubSpot Deal Score → Conversion Probability (historical)")
    left, right = st.columns([2, 1])
    with left:
        if not grp.empty:
            base = alt.Chart(grp).encode(x=alt.X("Range:N", sort=list(grp["Range"])))
            bars = base.mark_bar(opacity=0.9).encode(
                y=alt.Y("Total:Q", title="Count"),
                tooltip=["Range:N","Total:Q","Conversions:Q","Prob%:Q"]
            )
            line = base.mark_line(point=True).encode(
                y=alt.Y("Prob%:Q", title="Conversion Rate (%)", axis=alt.Axis(titleColor="#1f77b4")),
                color=alt.value("#1f77b4")
            )
            st.altair_chart(
                alt.layer(bars, line).resolve_scale(y='independent').properties(
                    height=360, title=f"Learned bins (lookback={lookback} mo, ref age={ref_age_days}d)"
                ),
                use_container_width=True
            )
        else:
            st.info("Not enough data to learn a calibration curve.")
    with right:
        st.dataframe(grp[["Range","Total","Conversions","Prob%"]].sort_values("Range"), use_container_width=True)
        st.download_button(
            "Download bins CSV",
            grp[["Range","Total","Conversions","Prob%"]].to_csv(index=False).encode("utf-8"),
            "hubspot_deal_score_bins.csv","text/csv", key="dl_hs_bins"
        )

    st.markdown("---")

    # ---------- Predict current-month likelihoods ----------
    st.markdown("### Running-month: normalized score → probability & expected conversions")

    cur_mask = dfm["__C"].dt.date.between(cur_start, cur_end) & has_score
    cur_df = dfm.loc[cur_mask, ["__C","__S"]].copy()
    if cur_df.empty:
        st.info("No deals created in the selected prediction window with a HubSpot Deal Score.")
        st.stop()

    cur_age = (pd.Timestamp(today_d) - cur_df["__C"]).dt.days.clip(lower=1)
    cur_df["__S_adj"] = cur_df["__S"] * np.minimum(ref_age_days / cur_age, 1.0)
    cur_df["__bin"] = pd.cut(cur_df["__S_adj"], bins=edges, include_lowest=True, right=True)

    cur_df = cur_df.merge(grp[["__bin","Prob%"]], on="__bin", how="left")
    cur_df["Prob%"] = cur_df["Prob%"].fillna(method="ffill").fillna(method="bfill")
    cur_df["Prob%"] = cur_df["Prob%"].fillna(float(grp["Prob%"].mean() if not grp["Prob%"].empty else 0.0))
    cur_df["Prob"] = cur_df["Prob%"] / 100.0

    expected_conversions = float(cur_df["Prob"].sum())
    total_deals = int(len(cur_df))

    k1, k2, k3 = st.columns(3)
    k1.metric("Deals in window", f"{total_deals:,}", help=f"{cur_start} → {cur_end}")
    k2.metric("Expected conversions (E[∑p])", f"{expected_conversions:.1f}")
    k3.metric("Avg probability", f"{(cur_df['Prob'].mean()*100.0):.1f}%")

    present = (cur_df.groupby("__bin").size().rename("Count").reset_index()
                      .merge(grp[["__bin","Prob%"]], on="__bin", how="left"))
    present["Range"] = present["__bin"].astype(str)
    st.altair_chart(
        alt.Chart(present).mark_bar(opacity=0.9).encode(
            x=alt.X("Range:N", sort=list(grp["Range"]), title="Score range (normalized)"),
            y=alt.Y("Count:Q"),
            tooltip=["Range:N","Count:Q","Prob%:Q"]
        ).properties(height=320, title="Current window — deal count by normalized score bin"),
        use_container_width=True
    )

    with st.expander("Download current-window probabilities"):
        out = cur_df[["__C","__S","__S_adj","Prob%"]].rename(columns={
            "__C":"Create Date", "__S":"HubSpot Deal Score", "__S_adj":f"Score (normalized to {ref_age_days}d)", "Prob%":"Estimated Conversion %"
        })
        st.dataframe(out.head(1000), use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                           "hubspot_deal_score_current_window_probs.csv","text/csv", key="dl_hs_cur")

    st.caption(
        "Notes: Normalization multiplies young-deal scores by min(ref_age / age, 1). "
        "Calibration uses historical lookback (excluding current month) with Laplace smoothing."
    )


# ===========================
# 📞 Performance ▸ Call Talk-time Report (non-intrusive add-on) — DDMMYYYY date parsing
# ===========================
import streamlit as st  # safe re-import
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, date, time

# ---- Date & Time & Duration parsers ----
def _ctt_parse_duration_hms(val) -> int:
    """Convert Call Duration to seconds.
    Accepts 'HH:MM:SS', 'HH;MM;SS', 'H:M:S', also 'MM:SS' or 'SS'.
    Returns 0 when missing/invalid."""
    if pd.isna(val):
        return 0
    s = str(val).strip().replace(';', ':')
    parts = [p for p in s.split(':') if p != '']
    try:
        if len(parts) == 3:
            hh, mm, ss = (int(float(p)) for p in parts)
        elif len(parts) == 2:
            hh, mm, ss = 0, int(float(parts[0])), int(float(parts[1]))
        elif len(parts) == 1:
            hh, mm, ss = 0, 0, int(float(parts[0]))
        else:
            return 0
        return hh*3600 + mm*60 + ss
    except Exception:
        return 0

_TIME_FORMATS = [
    "%I:%M:%S %p", "%I:%M %p", "%I %p",      # 12h with AM/PM
    "%H:%M:%S", "%H:%M", "%H",               # 24h
]

def _ctt_parse_time_only(val) -> time | None:
    """Accepts time like '18' (=> 18:00:00), '5;26;00 PM', '17:45:30', '09:05 AM'."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace(';', ':').upper().replace("  ", " ")
    try:
        if s.isdigit():
            h = int(s)
            if 0 <= h <= 23:
                return time(h, 0, 0)
    except Exception:
        pass
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    try:
        t = pd.to_datetime(s, errors="coerce").time()
        return t
    except Exception:
        return None

def _ctt_parse_date_ddmmyyyy(val) -> date | None:
    """Parse date strings specifically in DDMMYYYY (no separators) or with common separators (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY).
       Also tolerates Excel serials (coerced via pandas)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Handle pure 8-digit DDMMYYYY
    if s.isdigit() and len(s) == 8:
        try:
            return datetime.strptime(s, "%d%m%Y").date()
        except Exception:
            pass
    # Handle with separators
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %m %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # Fall back to pandas (dayfirst=True to bias DD/MM/YYYY)
    try:
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.notna(d):
            return d.date()
    except Exception:
        return None
    return None

def _ctt_build_dt(df: pd.DataFrame, date_col="Date", time_col="Time") -> pd.DataFrame:
    """Create a single datetime column _dt from separate Date (DDMMYYYY) and Time columns."""
    # Parse Date with custom DDMMYYYY logic
    d_series = df[date_col].apply(_ctt_parse_date_ddmmyyyy)
    # Parse Time using robust routine
    t_series = df[time_col].apply(_ctt_parse_time_only)
    # Build datetime; drop invalid
    dt = pd.to_datetime(pd.Series(d_series).astype(str) + " " + pd.Series(t_series).astype(str), errors="coerce")
    out = df.assign(_dt=dt).dropna(subset=["_dt"])
    return out

def _ctt_seconds_to_hms(total_seconds: int) -> str:
    total_seconds = int(total_seconds or 0)
    hh = total_seconds // 3600
    rem = total_seconds % 3600
    mm = rem // 60
    ss = rem % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _ctt_pick(df, preferred, cands):
    if preferred and preferred in df.columns: return preferred
    for c in cands:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def _render_call_talktime_report():
    st.subheader("Performance — Call Talk-time Report")
    st.caption("Date is parsed as DDMMYYYY (e.g., 25092025). Time supports formats like '18' (6 PM), '5;26;00 PM', '17:45:30'. Call Duration sums HH:MM:SS or HH;MM;SS.")

    upl = st.file_uploader("Upload activity feed CSV", type=["csv"], key="ctt_upl")
    if upl is None:
        st.info("Please upload the activityFeedReport_*.csv to continue.")
        return
    try:
        df_raw = pd.read_csv(upl)
    except Exception:
        text = upl.read().decode("utf-8", errors="ignore")
        df_raw = pd.read_csv(StringIO(text))

    # Resolve columns
    date_col     = _ctt_pick(df_raw, None, ["Date","Call Date"])
    time_col     = _ctt_pick(df_raw, None, ["Time","Call Time"])
    caller_col   = _ctt_pick(df_raw, None, ["Caller","Agent","Counsellor","Counselor","Caller Name","User"])
    type_col     = _ctt_pick(df_raw, None, ["Call Type","Type"])
    country_col  = _ctt_pick(df_raw, None, ["Country Name","Country"])
    duration_col = _ctt_pick(df_raw, None, ["Call Duration","Duration","Talk Time"])

    if any(x is None for x in [date_col, time_col, caller_col, duration_col]):
        st.error("Missing required columns. Need Date, Time, Caller, Call Duration.")
        return

    # Build _dt and seconds from duration
    df = df_raw.copy()
    df = _ctt_build_dt(df, date_col=date_col, time_col=time_col)
    if df.empty:
        st.warning("No valid rows after parsing Date (DDMMYYYY) & Time.")
        return
    df["_secs"] = df[duration_col].apply(_ctt_parse_duration_hms)

    # --- Filters: Date + Time ONLY ---
    min_d = df["_dt"].dt.date.min()
    max_d = df["_dt"].dt.date.max()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d_start = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="ctt_start_d")
    with c2:
        d_end = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d, key="ctt_end_d")
    with c3:
        t_start = st.time_input("Start time", value=time(0,0,0), key="ctt_start_t")
    with c4:
        t_end = st.time_input("End time", value=time(23,59,59), key="ctt_end_t")

    # Apply boundary filter via Date + Time only
    df_win = df[df["_dt"].dt.date.between(d_start, d_end)].copy()
    tt = df_win["_dt"].dt.time
    if t_start <= t_end:
        time_mask = (tt >= t_start) & (tt <= t_end)
    else:
        time_mask = (tt >= t_start) | (tt <= t_end)
    df_win = df_win[time_mask]

    # Optional secondary pickers
    callers = sorted(df_win[caller_col].dropna().astype(str).unique().tolist())
    types = sorted(df_win[type_col].dropna().astype(str).unique().tolist()) if type_col else []
    countries = sorted(df_win[country_col].dropna().astype(str).unique().tolist()) if country_col else []

    cA, cB, cC = st.columns(3)
    with cA:
        sel_callers = st.multiselect("Caller(s)", callers, default=callers, key="ctt_callers")
    with cB:
        sel_types = st.multiselect("Call Type(s)", types, default=types, key="ctt_types") if types else []
    with cC:
        sel_countries = st.multiselect("Country Name(s)", countries, default=countries, key="ctt_ctys") if countries else []

    mask = df_win[caller_col].astype(str).isin(sel_callers)
    if type_col and sel_types:
        mask &= df_win[type_col].astype(str).isin(sel_types)
    if country_col and sel_countries:
        mask &= df_win[country_col].astype(str).isin(sel_countries)
    df_win = df_win[mask].copy()
    if df_win.empty:
        st.warning("No rows after applying filters.")
        return

    # KPIs
    total_secs = int(df_win["_secs"].sum())
    total_calls = int(len(df_win))
    avg_secs = int(round(df_win["_secs"].mean())) if total_calls else 0
    med_secs = int(df_win["_secs"].median()) if total_calls else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Talk Time", _ctt_seconds_to_hms(total_secs))
    k2.metric("# Calls", f"{total_calls:,}")
    k3.metric("Avg Call Duration", _ctt_seconds_to_hms(avg_secs))
    k4.metric("Median Call Duration", _ctt_seconds_to_hms(med_secs))

    # Aggregations
    caller_tot = (df_win.groupby(caller_col, dropna=False)["_secs"].sum().reset_index()
                  .rename(columns={"_secs":"Total Seconds"}).sort_values("Total Seconds", ascending=False))
    caller_tot["Total Talk Time"] = caller_tot["Total Seconds"].map(_ctt_seconds_to_hms)

    gt60 = df_win[df_win["_secs"] > 60]
    caller_tot_gt60 = (gt60.groupby(caller_col, dropna=False)["_secs"].sum().reset_index()
                       .rename(columns={"_secs":"Total Seconds (>60s)"}).sort_values("Total Seconds (>60s)", ascending=False))
    caller_tot_gt60["Total Talk Time (>60s)"] = caller_tot_gt60["Total Seconds (>60s)"].map(_ctt_seconds_to_hms)

    if country_col:
        country_tot = (df_win.groupby(country_col, dropna=False)["_secs"].sum().reset_index()
                       .rename(columns={"_secs":"Total Seconds"}).sort_values("Total Seconds", ascending=False))
        country_tot["Total Talk Time"] = country_tot["Total Seconds"].map(_ctt_seconds_to_hms)
        country_tot_gt60 = (gt60.groupby(country_col, dropna=False)["_secs"].sum().reset_index()
                            .rename(columns={"_secs":"Total Seconds (>60s)"}).sort_values("Total Seconds (>60s)", ascending=False))
        country_tot_gt60["Total Talk Time (>60s)"] = country_tot_gt60["Total Seconds (>60s)"].map(_ctt_seconds_to_hms)
    else:
        country_tot = pd.DataFrame(columns=["Country","Total Seconds","Total Talk Time"])
        country_tot_gt60 = pd.DataFrame(columns=["Country","Total Seconds (>60s)","Total Talk Time (>60s)"])

    st.markdown("### 1) Caller wise — Total Call Duration")
    st.dataframe(caller_tot[[caller_col, "Total Talk Time", "Total Seconds"]], use_container_width=True)
    st.download_button("⬇️ Download Caller Totals (All)", caller_tot.to_csv(index=False).encode("utf-8"), "caller_total_all.csv", "text/csv")

    st.markdown("### 2) Caller wise — Total Call Duration (> 60 sec)")
    st.dataframe(caller_tot_gt60[[caller_col, "Total Talk Time (>60s)", "Total Seconds (>60s)"]], use_container_width=True)
    st.download_button("⬇️ Download Caller Totals (>60s)", caller_tot_gt60.to_csv(index=False).encode("utf-8"), "caller_total_gt60.csv", "text/csv")

    if country_col:
        st.markdown("### 3) Country wise — Total Call Duration")
        st.dataframe(country_tot[[country_col, "Total Talk Time", "Total Seconds"]], use_container_width=True)
        st.download_button("⬇️ Download Country Totals (All)", country_tot.to_csv(index=False).encode("utf-8"), "country_total_all.csv", "text/csv")

        st.markdown("### 4) Country wise — Total Call Duration (> 60 sec)")
        st.dataframe(country_tot_gt60[[country_col, "Total Talk Time (>60s)", "Total Seconds (>60s)"]], use_container_width=True)
        st.download_button("⬇️ Download Country Totals (>60s)", country_tot_gt60.to_csv(index=False).encode("utf-8"), "country_total_gt60.csv", "text/csv")
    else:
        st.info("Country column not found — skipping country-wise breakdowns.")

    # 5) Caller-wise calling journey
    st.markdown("### 5) Caller-wise Calling Journey (Hour-of-Day)")
    df_win = df_win.assign(_hour=df_win["_dt"].dt.hour)
    per_caller_hour = (df_win.groupby([caller_col, "_hour"], dropna=False)["_secs"].sum().reset_index()
                       .rename(columns={"_secs":"Total Seconds"}))

    def _agg_max_min(g):
        if g.empty:
            return pd.Series({"Max Hour": np.nan, "Max Hour Talk Time": 0, "Min Hour": np.nan, "Min Hour Talk Time": 0})
        g = g.sort_values("Total Seconds", ascending=False)
        max_hour = int(g.iloc[0]["_hour"]); max_val = int(g.iloc[0]["Total Seconds"])
        g_nonzero = g[g["Total Seconds"] > 0]
        g_min = (g_nonzero if not g_nonzero.empty else g).sort_values("Total Seconds", ascending=True).iloc[0]
        min_hour = int(g_min["_hour"]); min_val = int(g_min["Total Seconds"])
        return pd.Series({"Max Hour": max_hour, "Max Hour Talk Time": max_val, "Min Hour": min_hour, "Min Hour Talk Time": min_val})

    caller_hour_summary = per_caller_hour.groupby(caller_col, dropna=False).apply(_agg_max_min).reset_index()
    caller_hour_summary["Max Hour Talk Time (HH:MM:SS)"] = caller_hour_summary["Max Hour Talk Time"].map(_ctt_seconds_to_hms)
    caller_hour_summary["Min Hour Talk Time (HH:MM:SS)"] = caller_hour_summary["Min Hour Talk Time"].map(_ctt_seconds_to_hms)

    st.markdown("**Max/Min Hour per Caller (by Talk-time)**")
    st.dataframe(caller_hour_summary[[caller_col, "Max Hour", "Max Hour Talk Time (HH:MM:SS)", "Min Hour", "Min Hour Talk Time (HH:MM:SS)"]], use_container_width=True)
    st.download_button("⬇️ Download Caller Hour Summary", caller_hour_summary.to_csv(index=False).encode("utf-8"), "caller_hour_summary.csv", "text/csv")

    focus = st.selectbox("Focus: Caller hour profile", ["(All)"] + callers, index=0, key="ctt_focus")
    if focus != "(All)":
        foc = per_caller_hour[per_caller_hour[caller_col] == focus].sort_values("_hour")
        foc["Talk Time (HH:MM:SS)"] = foc["Total Seconds"].map(_ctt_seconds_to_hms)
        st.markdown(f"**Hourly Distribution for `{focus}`**")
        st.dataframe(foc[["_hour", "Talk Time (HH:MM:SS)", "Total Seconds"]], use_container_width=True)
        st.download_button(f"⬇️ Download Hourly Profile — {focus}", foc.to_csv(index=False).encode("utf-8"), f"hourly_profile_{focus}.csv", "text/csv")

# Router: only runs this add-on when the new pill is selected; otherwise no effect
try:
    if 'view' in globals() and view == "Call Talk-time Report":
        _render_call_talktime_report()
except Exception:
    pass




# --- Performance — Original source ---
def _render_performance_original_source(
    df_f,
    create_col: str | None,
    enrol_col: str | None,
    drill1_col: str | None,
    drill2_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import altair as alt
    from datetime import date, timedelta

    st.subheader("Performance — Original source")

    # Column picks
    def _pick(df, *cands):
        for c in cands:
            if c and c in df.columns: return c
        return None

    create_col = _pick(df_f, create_col, "Create Date","Create date","Create_Date","Created At","Deal Create Date","Created On")
    enrol_col  = _pick(df_f, enrol_col, "Payment Received Date","Enrolment Date","Enrollment Date","Payment Date","Paid On")
    drill1_col = _pick(df_f, drill1_col, "Original Traffic Source Drill-Down 1","Traffic Source Drill-Down 1","OTS DD1")
    drill2_col = _pick(df_f, drill2_col, "Original Traffic Source Drill-Down 2","Traffic Source Drill-Down 2","OTS DD2")

    if not (drill1_col or drill2_col):
        st.warning("Could not find **Original Traffic Source Drill-Down 1/2** columns.")
        return
    if not create_col:
        st.warning("Could not find **Create Date** column.")
        return

    # Controls
    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="os_mode")
    measure = st.radio("Count basis", ["Deals Created (Create Date)","Enrollments (Payment Received Date)"], index=0, horizontal=True, key="os_measure")
    level = st.selectbox("Drilldown level", ["Original Traffic Source Drill-Down 1","Original Traffic Source Drill-Down 2"], index=0, key="os_level")

    today = date.today()
    lbl = "Create Date" if measure.startswith("Deals") else "Payment Received Date"
    preset = st.radio(f"Date range ({lbl})", ["Today","Yesterday","This Month","Custom"], index=2, horizontal=True, key="os_rng")
    if preset == "Today":
        start, end = today, today
    elif preset == "Yesterday":
        start = today - timedelta(days=1); end = start
    elif preset == "This Month":
        start = today.replace(day=1); end = today
    else:
        c1, c2 = st.columns(2)
        with c1: start = st.date_input("Start", value=today.replace(day=1), key="os_start")
        with c2: end   = st.date_input("End", value=today, key="os_end")
        if start > end: start, end = end, start

    # Working frame
    df = df_f.copy()
    def _to_dt(s):
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)

    df["_create"] = _to_dt(df[create_col]) if create_col in df.columns else pd.NaT
    df["_enrol"]  = _to_dt(df[enrol_col])  if enrol_col  and enrol_col in df.columns else pd.NaT
    df["_d1"] = df[drill1_col] if drill1_col in df.columns else None
    df["_d2"] = df[drill2_col] if drill2_col in df.columns else None

    # Choose basis
    if measure.startswith("Deals"):
        df["_basis"] = df["_create"]
    else:
        if not enrol_col:
            st.info("No enrollment date column found; falling back to **Create Date** basis.")
            df["_basis"] = df["_create"]
        else:
            df["_basis"] = df["_enrol"]

    # Filter by window
    mask = df["_basis"].dt.date.between(start, end)
    if mode == "MTD":
        # For MTD, also require Create-Date month/year == selected month/year
        df["_cohort_ok"] = (df["_create"].dt.month == start.month) & (df["_create"].dt.year == start.year)
        mask = mask & df["_cohort_ok"].fillna(False)

    df = df.loc[mask].copy()
    if df.empty:
        st.info("No rows in the selected window/filters."); return


    # X-axis option
    x_axis = st.selectbox(
        "X-axis",
        ["Date (daily)", "Original Traffic Source Drill-Down 1", "Original Traffic Source Drill-Down 2"],
        index=0,
        key="os_xaxis"
    )
    # Limit categories shown (Top-N)
    if "os_topn" not in st.session_state:
        st.session_state["os_topn"] = 10
    cA, cB, cC, cD = st.columns([0.7,0.7,3,1.2])
    with cA:
        if st.button("–", key="os_dec"):
            st.session_state["os_topn"] = max(1, int(st.session_state.get("os_topn",10)) - 1)
    with cB:
        if st.button("+", key="os_inc"):
            st.session_state["os_topn"] = int(st.session_state.get("os_topn",10)) + 1
    with cC:
        st.session_state["os_topn"] = int(st.number_input("Show top N categories", min_value=1, max_value=200, value=int(st.session_state.get("os_topn",10)), step=1, key="os_topn_input"))
    with cD:
        if st.button("Top 10", key="os_top10"):
            st.session_state["os_topn"] = 10
    topN = int(st.session_state.get("os_topn", 10))

        
    # Daily distribution (prepare)
    df["_day"] = df["_basis"].dt.date
    level_col = drill1_col if level.endswith("Down 1") else drill2_col
    if not level_col or level_col not in df.columns:
        st.warning("Selected drilldown level column not found."); return

    if x_axis == "Date (daily)":
        # Date on X; color = selected drilldown level
        grp = df.groupby(["_day", level_col], dropna=False).size().reset_index(name="Count")
        # Clean category values to avoid NaN in column names
        grp[level_col] = grp[level_col].astype(str)
        grp[level_col] = grp[level_col].replace({'nan':'Unknown','None':'Unknown','NaT':'Unknown'})
        # Limit to Top-N categories by total across window
        top_cats = (grp.groupby(level_col)["Count"].sum().sort_values(ascending=False).head(topN).index.tolist())
        grp = grp[grp[level_col].isin(top_cats)].copy()
        pivot = grp.pivot(index="_day", columns=level_col, values="Count").fillna(0).reset_index()
        # Ensure pivot columns are strings (no NaN column names)
        pivot.columns = [("Unknown" if (isinstance(c,float) and c != c) else str(c)) for c in pivot.columns]

        # KPIs
        total = int(grp["Count"].sum()); days = grp["_day"].nunique()
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Total", f"{total:,}")
        with c2: st.metric("Days", days)
        with c3: st.metric("Avg / day", f"{(total/days if days else 0):.1f}")

        # Graph
        chart_df = grp.copy()
        chart_df[level_col] = chart_df[level_col].astype(str)
        chart_df[level_col] = chart_df[level_col].replace({'nan':"Unknown", "None":"Unknown", "NaT":"Unknown"})
        chart_type = st.selectbox("Graph type", ["Stacked Bar","Bar","Line"], index=0, key="os_chart")
        base = alt.Chart(chart_df).encode(
            x=alt.X("_day:T", title="Date"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(f"{level_col}:N", title=level_col),
            tooltip=[alt.Tooltip("_day:T", title="Date"), alt.Tooltip(f"{level_col}:N", title=level_col), alt.Tooltip("Count:Q")]
        ).properties(height=360)

        if chart_type == "Line":
            ch = base.mark_line(point=True)
        elif chart_type == "Bar":
            ch = base.mark_bar()
        else:
            ch = base.mark_bar()  # Stacked
        st.altair_chart(ch, use_container_width=True)

        # Table + download
        st.dataframe(pivot, use_container_width=True, hide_index=True)
        csv = pivot.to_csv(index=False).encode("utf-8")
        st.download_button("Download table (CSV)", data=csv, file_name="original_source_distribution_by_day.csv", mime="text/csv", key="os_dl_byday")

    else:
        # Drilldown on X: aggregate totals in window, optionally for chosen drilldown field
        if x_axis.endswith("Down 1"):
            xcol = drill1_col
        else:
            xcol = drill2_col

        grp = df.groupby(xcol, dropna=False).size().reset_index(name="Count")
        grp[xcol] = grp[xcol].astype(str)
        grp[xcol] = grp[xcol].replace({'nan':'Unknown','None':'Unknown','NaT':'Unknown'})
        grp = grp.sort_values("Count", ascending=False).head(topN).reset_index(drop=True)
        
        total = int(grp["Count"].sum())
        c1,c2 = st.columns(2)
        with c1: st.metric("Total", f"{total:,}")
        with c2: st.metric("Categories", grp.shape[0])

        chart_type = st.selectbox("Graph type", ["Bar","Line"], index=0, key="os_chart_xcat")
        base = alt.Chart(grp).encode(
            x=alt.X(f"{xcol}:N", sort='-y', title=xcol),
            y=alt.Y("Count:Q", title="Count"),
            tooltip=[alt.Tooltip(f"{xcol}:N", title=xcol), alt.Tooltip("Count:Q")]
        ).properties(height=360)

        ch = base.mark_bar() if chart_type == "Bar" else base.mark_line(point=True)
        st.altair_chart(ch, use_container_width=True)

        # Table + download
        tbl = grp.rename(columns={xcol: "Original Source Category"})
        tbl.columns = [("Unknown" if (isinstance(c,float) and c != c) else str(c)) for c in tbl.columns]
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        csv = grp.to_csv(index=False).encode("utf-8")
        st.download_button("Download table (CSV)", data=csv, file_name="original_source_distribution_by_category.csv", mime="text/csv", key="os_dl_bycat")
# Table + download
    st.dataframe(pivot, use_container_width=True, hide_index=True)
    csv = pivot.to_csv(index=False).encode("utf-8")
    st.download_button("Download table (CSV)", data=csv, file_name="original_source_distribution.csv", mime="text/csv", key="os_dl")


def _render_performance_quick_view(
    df_f: pd.DataFrame,
    *,
    create_col: str,
    pay_col: str,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    cal_done_col: str | None,
    source_col: str | None
):
    import streamlit as st
    import pandas as pd
    from datetime import date, timedelta
    import altair as alt

    st.subheader("Performance — Quick View")

    # ---- helpers to resolve columns robustly
    def _pick_col(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # simple case-insensitive pass
        lc = {c.lower().strip(): c for c in df.columns}
        for c in candidates:
            key = c.lower().strip()
            if key in lc:
                return lc[key]
        return None

    ac_candidates = ["Academic Counsellor","AC","Counsellor","Sales Owner","Lead Owner","Owner","Advisor","Agent"]
    country_candidates = ["Country","Country Name","Geo","Region"]

    ac_col_res = _pick_col(df_f, ac_candidates)
    country_col_res = _pick_col(df_f, country_candidates)

    # ---- column guards
    if source_col is None or source_col not in df_f.columns:
        src_series = pd.Series(["Unknown"] * len(df_f), index=df_f.index, name="__src__")
    else:
        src_series = df_f[source_col].fillna("Unknown").astype(str).rename("__src__")

    ac_series = df_f[ac_col_res].fillna("Unknown").astype(str).rename("__ac__") if ac_col_res else pd.Series(["Unknown"] * len(df_f), index=df_f.index, name="__ac__")
    country_series = df_f[country_col_res].fillna("Unknown").astype(str).rename("__country__") if country_col_res else pd.Series(["Unknown"] * len(df_f), index=df_f.index, name="__country__")

    # Normalize date-like columns (dayfirst safe)
    def _dt(s):
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)

    d = df_f.copy()
    d["__create"] = _dt(d[create_col])
    d["__pay"]    = _dt(d[pay_col])
    d["__src"]    = src_series
    d["__ac"]     = ac_series
    d["__country"]= country_series

    # Optional calibration dates
    _first  = _dt(d[first_cal_sched_col]) if (first_cal_sched_col and first_cal_sched_col in d.columns) else pd.NaT
    _resch  = _dt(d[cal_resched_col])     if (cal_resched_col     and cal_resched_col     in d.columns) else pd.NaT
    _done   = _dt(d[cal_done_col])        if (cal_done_col        and cal_done_col        in d.columns) else pd.NaT
    d["__first"] = _first
    d["__resch"] = _resch
    d["__done"]  = _done

    today = date.today()
    yday  = today - timedelta(days=1)

    # ----- build masks and three working DataFrames for yesterday/today/range
    def _prep_window(start_d, end_d, mode=None):
        m_create = d["__create"].dt.date.between(start_d, end_d)
        m_pay    = d["__pay"].dt.date.between(start_d, end_d)
        m_first  = pd.to_datetime(d["__first"], errors="coerce").dt.date.between(start_d, end_d) if d["__first"].notna().any() else pd.Series(False, index=d.index)
        m_resch  = pd.to_datetime(d["__resch"], errors="coerce").dt.date.between(start_d, end_d) if d["__resch"].notna().any() else pd.Series(False, index=d.index)
        m_done   = pd.to_datetime(d["__done"],  errors="coerce").dt.date.between(start_d, end_d)  if d["__done"].notna().any()  else pd.Series(False, index=d.index)

        if mode == "MTD":
            enrol_mask = m_pay & m_create
            first_mask = m_first & m_create
            resch_mask = m_resch & m_create
            done_mask  = m_done  & m_create
        else:
            enrol_mask = m_pay
            first_mask = m_first
            resch_mask = m_resch
            done_mask  = m_done

        dfw = d.copy()
        dfw["Deals_Created"] = m_create.astype(int)
        dfw["First_Cal"]     = first_mask.astype(int)
        dfw["Cal_Resched"]   = resch_mask.astype(int)
        dfw["Cal_Done"]      = done_mask.astype(int)
        dfw["Enrolments"]    = enrol_mask.astype(int)
        return dfw

    # Yesterday / Today
    d_y = _prep_window(yday, yday, mode=None)
    d_t = _prep_window(today, today, mode=None)

    # Range controls + mode
    st.markdown("### Range — MTD/Cohort")
    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="qv_mode")
    cA, cB, cC = st.columns(3)
    with cA:
        rng_start = st.date_input("Start date", value=date.today().replace(day=1), key="qv_start")
    with cB:
        rng_end = st.date_input("End date", value=date.today(), key="qv_end")
    with cC:
        st.caption("MTD = Pay in range **and** Create in range • Cohort = Pay in range")

    if rng_end < rng_start:
        st.error("End date cannot be before start date.")
        return

    d_r = _prep_window(rng_start, rng_end, mode=mode)

    # ---- generic table section
    def _section_tables(df_source: pd.DataFrame, title: str, group_key: str | None, label_suffix: str, display_name: str | None = None):
        with st.container():
            st.markdown(title)
            c1, c2, c3 = st.columns(3)
            if group_key is None:
                with c1:
                    st.dataframe(pd.DataFrame({f"Deals Created {label_suffix}":[int(df_source['Deals_Created'].sum())]}),
                                 use_container_width=True, hide_index=True)
                with c2:
                    st.dataframe(pd.DataFrame([{
                        f"First Cal {label_suffix}": int(df_source["First_Cal"].sum()),
                        f"Cal Resch {label_suffix}": int(df_source["Cal_Resched"].sum()),
                        f"Cal Done {label_suffix}" : int(df_source["Cal_Done"].sum()),
                    }]), use_container_width=True, hide_index=True)
                with c3:
                    st.dataframe(pd.DataFrame({f"Enrolments {label_suffix}":[int(df_source['Enrolments'].sum())]}),
                                 use_container_width=True, hide_index=True)
            else:
                label = display_name if display_name else group_key.strip("_").strip("__").title().replace("_"," ")
                with c1:
                    bx1 = (
                        df_source.groupby(group_key, dropna=False)["Deals_Created"]
                                 .sum()
                                 .reset_index()
                                 .rename(columns={group_key: label, "Deals_Created": f"Deals Created {label_suffix}"})
                    )
                    st.dataframe(bx1, use_container_width=True, hide_index=True)
                with c2:
                    bx2 = (
                        df_source.groupby(group_key, dropna=False)[["First_Cal","Cal_Resched","Cal_Done"]]
                                 .sum()
                                 .reset_index()
                                 .rename(columns={group_key: label,
                                                  "First_Cal": f"First Cal {label_suffix}",
                                                  "Cal_Resched": f"Cal Resch {label_suffix}",
                                                  "Cal_Done": f"Cal Done {label_suffix}"})
                    )
                    st.dataframe(bx2, use_container_width=True, hide_index=True)
                with c3:
                    bx3 = (
                        df_source.groupby(group_key, dropna=False)["Enrolments"]
                                 .sum()
                                 .reset_index()
                                 .rename(columns={group_key: label, "Enrolments": f"Enrolments {label_suffix}"})
                    )
                    st.dataframe(bx3, use_container_width=True, hide_index=True)

    # Tabs
    tab_src, tab_ac, tab_ctry, tab_overall = st.tabs(["By Source", "By Academic Counsellor", "By Country", "Overall"])

    with tab_src:
        _section_tables(d_y, "## Yesterday", group_key="__src", label_suffix="(yesterday)", display_name="JetLearn Deal Source")
        st.divider()
        _section_tables(d_t, "## Today", group_key="__src", label_suffix="(today)", display_name="JetLearn Deal Source")
        st.divider()
        _section_tables(d_r, "## Range", group_key="__src", label_suffix="(range)", display_name="JetLearn Deal Source")

    with tab_ac:
        _section_tables(d_y, "## Yesterday", group_key="__ac", label_suffix="(yesterday)", display_name=ac_col_res or "Academic Counsellor")
        st.divider()
        _section_tables(d_t, "## Today", group_key="__ac", label_suffix="(today)", display_name=ac_col_res or "Academic Counsellor")
        st.divider()
        _section_tables(d_r, "## Range", group_key="__ac", label_suffix="(range)", display_name=ac_col_res or "Academic Counsellor")

    with tab_ctry:
        _section_tables(d_y, "## Yesterday", group_key="__country", label_suffix="(yesterday)", display_name=country_col_res or "Country")
        st.divider()
        _section_tables(d_t, "## Today", group_key="__country", label_suffix="(today)", display_name=country_col_res or "Country")
        st.divider()
        _section_tables(d_r, "## Range", group_key="__country", label_suffix="(range)", display_name=country_col_res or "Country")

    with tab_overall:
        _section_tables(d_y, "## Yesterday — Overall", group_key=None, label_suffix="(yesterday)")
        st.divider()
        _section_tables(d_t, "## Today — Overall", group_key=None, label_suffix="(today)")
        st.divider()
        _section_tables(d_r, "## Range — Overall", group_key=None, label_suffix="(range)")

    st.divider()

    # ---------- (D) Combined graph: Daily Deals vs Enrolments with "Exclude Referral" toggle ----------
    st.markdown("## Daily: Deals (bars) vs Enrolments (line)")
    exclude_ref = st.checkbox("Exclude Referral from Deals", value=False, key="qv_ex_ref")
    deal_src = d.copy()
    if exclude_ref:
        deal_src = deal_src[~deal_src["__src"].str.contains("referr", case=False, na=False)]

    base_start = rng_start
    base_end   = rng_end
    days = pd.date_range(base_start, base_end, freq="D").date

    # Deals series based on source (referral-exclusion applies only to deals)
    d_series = (
        deal_src[deal_src["__create"].dt.date.between(base_start, base_end)]
        .groupby(deal_src["__create"].dt.date)
        .size()
        .reindex(days, fill_value=0)
        .rename("Deals")
    )

    if mode == "MTD":
        enrol_line_mask = d["__pay"].dt.date.between(base_start, base_end) & d["__create"].dt.date.between(base_start, base_end)
    else:
        enrol_line_mask = d["__pay"].dt.date.between(base_start, base_end)

    e_series = (
        d.loc[enrol_line_mask]
         .groupby(d["__pay"].dt.date)
         .size()
         .reindex(days, fill_value=0)
         .rename("Enrolments")
    )

    ts = pd.concat([d_series, e_series], axis=1).reset_index().rename(columns={"index":"Date"})

    base = alt.Chart(ts).encode(x=alt.X("Date:T", axis=alt.Axis(title=None)))
    bars = base.mark_bar(opacity=0.8).encode(
        y=alt.Y("Deals:Q", axis=alt.Axis(title="Deals")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Deals:Q")]
    ).properties(height=280)
    line = base.mark_line(point=True).encode(
        y=alt.Y("Enrolments:Q", axis=alt.Axis(title="Enrolments")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Enrolments:Q")]
    )
    st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent').properties(title="Deals (bars) vs Enrolments (line)"), use_container_width=True)




# --- Performance / Sales Activity (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Sales Activity":
        _render_performance_sales_activity(
            df_f=df_f,
            first_cal_sched_col=first_cal_sched_col if 'first_cal_sched_col' in globals() else None,
            cal_resched_col=cal_resched_col if 'cal_resched_col' in globals() else None,
            slot_col=calibration_slot_col if 'calibration_slot_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Sales Activity error: {_e}")
    except Exception:
        pass


# --- Performance / Activity Tracker (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Activity Tracker":
        _render_performance_activity_tracker(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Activity Tracker error: {_e}")
    except Exception:
        pass





# ------------------------------
# Performance — Activity concentration
# ------------------------------
def _render_performance_activity_concentration(
    df_f,
    create_col: str | None,
    country_col: str | None,
    source_col: str | None,
    counsellor_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date, timedelta

    st.subheader("Performance — Activity concentration")

    def _pick(df: pd.DataFrame, *cands):
        for c in cands:
            if c and c in df.columns:
                return c
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if isinstance(c, str) and c.lower().strip() in low:
                return low[c.lower().strip()]
        return None

    # Resolve columns
    last_act_col = _pick(df_f, "Last Activity Date", "Last activity date", "Last_Activity_Date", "[Last] Activity Date")
    _create_col = create_col or _pick(df_f, "Create Date","Create date","Create_Date","Created At")
    _country_col = country_col or _pick(df_f, "Country")
    _source_col = source_col or _pick(df_f, "JetLearn Deal Source","Deal Source","Source")
    _counsellor_col = counsellor_col or _pick(df_f, "Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor")

    if not last_act_col:
        st.info("No **Last Activity Date** column detected — cannot render Activity concentration.")
        return
    if not _create_col:
        st.info("No **Create Date** column detected — cannot render Activity concentration.")
        return

    # Controls (unique keys with ac_ prefix)
    st.caption("Choose an **Activity Date** window to see which **Create Date cohorts** were worked on, and their mix.")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        preset = st.radio("Preset", ["Today","Yesterday","This Month","Custom"], index=1, key="ac_preset")
    today = date.today()
    if preset == "Today":
        start_d = end_d = today
    elif preset == "Yesterday":
        start_d = end_d = today - timedelta(days=1)
    elif preset == "This Month":
        start_d = date(today.year, today.month, 1)
        end_d = today
    else:
        with c2:
            start_d = st.date_input("Start", today - timedelta(days=7), key="ac_start")
        with c3:
            end_d = st.date_input("End", today, key="ac_end")
    if start_d > end_d:
        st.warning("Start date is after end date. Please adjust.")
        return

    # Convert dates
    def _to_dt(s):
        return pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False, infer_datetime_format=True)

    df = df_f.copy()
    df["_ac_last"] = _to_dt(df[last_act_col])
    df["_ac_create"] = _to_dt(df[_create_col])

    # Filter window on activity date (inclusive)
    mask = (df["_ac_last"].dt.date >= pd.to_datetime(start_d).date()) & (df["_ac_last"].dt.date <= pd.to_datetime(end_d).date())
    dfw = df.loc[mask].copy()
    total_in_window = len(dfw)
    if total_in_window == 0:
        st.info("No rows in the selected **Activity Date** window.")
        return

    # Cohort month for create date
    dfw["_cohort"] = dfw["_ac_create"].dt.to_period("M").astype(str)

    # Robust month-age computation (avoids MonthEnd offsets)
    _today = pd.Timestamp.today()
    _today_month_id = _today.year * 12 + _today.month
    _create_month_id = (dfw["_ac_create"].dt.year * 12 + dfw["_ac_create"].dt.month)
    dfw["_cohort_age_m"] = (_today_month_id - _create_month_id).astype("Int64").fillna(0)

    # Stack-by chooser
    stack_by = st.selectbox("Stack by", ["None","Country","Source","Counsellor"], index=0, key="ac_stack")
    if stack_by == "Country":
        stack_col = _country_col
    elif stack_by == "Source":
        stack_col = _source_col
    elif stack_by == "Counsellor":
        stack_col = _counsellor_col
    else:
        stack_col = None

    chart_type = st.selectbox("Chart type", ["Stacked Bar","Bar","Line"], index=0, key="ac_chart")

    # Prepare aggregation
    if stack_col and stack_col in dfw.columns:
        lvl = dfw[stack_col].fillna("Unknown")
        grp = dfw.groupby(["_cohort", lvl]).size().reset_index(name="count")
        grp.columns = ["cohort", "stack", "count"]
    else:
        grp = dfw.groupby(["_cohort"]).size().reset_index(name="count")
        grp["stack"] = "All"
        # Rename and order columns explicitly to avoid misalignment
        grp = grp.rename(columns={"_cohort": "cohort"})[["cohort", "stack", "count"]]

    # Sort cohorts chronologically
    try:
        grp["_cohort_dt"] = pd.to_datetime(grp["cohort"] + "-01", errors="coerce")
    except Exception:
        grp["_cohort_dt"] = pd.NaT
    grp = grp.sort_values("_cohort_dt")
    # Ensure count is numeric
    grp["count"] = pd.to_numeric(grp["count"], errors="coerce").fillna(0).astype(int)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Deals worked", f"{total_in_window:,}")
    with k2: st.metric("Cohorts touched", f"{grp['cohort'].nunique():,}")
    top_row = grp.groupby("cohort")["count"].sum().sort_values(ascending=False).head(1)
    top_label = top_row.index[0] if not top_row.empty else "—"
    top_val = int(top_row.iloc[0]) if not top_row.empty else 0
    with k3: st.metric("Top cohort", f"{top_label}", f"{top_val:,}")
    with k4:
        med_age = int(pd.to_numeric(dfw["_cohort_age_m"], errors="coerce").median(skipna=True) or 0) if len(dfw) else 0
        st.metric("Median cohort age (months)", f"{med_age}")

    # Chart
    base = alt.Chart(grp).encode(
        x=alt.X("cohort:N", sort=grp["cohort"].tolist(), title="Create Date cohort (YYYY-MM)"),
        y=alt.Y("count:Q", title="Deals worked"),
        tooltip=["cohort:N","count:Q"] + ([alt.Tooltip("stack:N", title=stack_by)] if stack_col else [])
    )
    if chart_type == "Line":
        ch = base.mark_line(point=True).encode(color="stack:N" if stack_col else alt.value("gray"))
    elif chart_type == "Bar":
        ch = base.mark_bar().encode(color="stack:N" if stack_col else alt.value("gray"))
    else:
        ch = base.mark_bar().encode(color="stack:N")
    st.altair_chart(ch, use_container_width=True)

    # Composition tables
    st.markdown("### Composition")
    colA, colB, colC = st.columns(3)
    # Top cohorts
    top_coh = grp.groupby("cohort")["count"].sum().reset_index().sort_values("count", ascending=False)
    top_coh["share_%"] = (top_coh["count"] / top_coh["count"].sum() * 100).round(1)
    with colA:
        st.markdown("**Cohorts**")
        st.dataframe(top_coh.head(25), use_container_width=True, hide_index=True)
    # Country mix
    if _country_col and _country_col in dfw.columns:
        mix_cty = dfw.groupby(_country_col).size().reset_index(name="count").sort_values("count", ascending=False)
        mix_cty["share_%"] = (mix_cty["count"]/mix_cty["count"].sum()*100).round(1)
        with colB:
            st.markdown("**Country mix**")
            st.dataframe(mix_cty.head(25), use_container_width=True, hide_index=True)
    # Source mix
    if _source_col and _source_col in dfw.columns:
        mix_src = dfw.groupby(_source_col).size().reset_index(name="count").sort_values("count", ascending=False)
        mix_src["share_%"] = (mix_src["count"]/mix_src["count"].sum()*100).round(1)
        with colC:
            st.markdown("**Deal source mix**")
            st.dataframe(mix_src.head(25), use_container_width=True, hide_index=True)

    # Counsellor mix (full table below)
    if _counsellor_col and _counsellor_col in dfw.columns:
        st.markdown("**Counsellor mix (full)**")
        mix_csl = dfw.groupby(_counsellor_col).size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(mix_csl, use_container_width=True, hide_index=True)

    # Detailed wide table for download
    if stack_col and stack_col in dfw.columns:
        pivot = dfw.pivot_table(index="_cohort", columns=stack_col, values="_ac_last", aggfunc="count", fill_value=0)
        pivot = pivot.reset_index().rename(columns={"_cohort":"cohort"})
    else:
        pivot = grp[["cohort","count"]].copy()
    st.download_button(
        "Download CSV — Activity concentration",
        (pivot.to_csv(index=False).encode("utf-8")),
        file_name="activity_concentration.csv",
        mime="text/csv",
        key="ac_dl_csv"
    )





# ------------------------------
# Performance — Lead mix (multi-window by Deal Source)
# ------------------------------

def _render_performance_lead_mix(
    df_f,
    create_col: str | None = None,
    pay_col: str | None = None,
    first_cal_sched_col: str | None = None,
    cal_resched_col: str | None = None,
    cal_done_col: str | None = None,
    source_col: str | None = None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date, timedelta

    st.subheader("Performance — Lead mix")

    # Mode toggle
    view_mode = st.radio("Mode", ["MTD","Cohort"], index=0, horizontal=True, key="lm_mode")
    view_mode = "Cohort" if view_mode == "MTD" else "MTD"

    # --- Guards ---
    if create_col is None or create_col not in df_f.columns:
        st.error("Create Date column not found."); return
    if pay_col is None or pay_col not in df_f.columns:
        st.error("Payment Received Date column not found."); return

    # --- Column helpers ---
    def _pick(df: pd.DataFrame, *cands):
        for c in cands:
            if c and c in df.columns:
                return c
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if isinstance(c, str) and c.lower().strip() in low:
                return low[c.lower().strip()]
        return None

    _source = source_col or _pick(df_f, "JetLearn Deal Source","Deal Source","Source")
    _country = _pick(df_f, "Country")
    _counsellor = _pick(df_f, "Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor")
    first_cal = first_cal_sched_col or _pick(df_f, "First Calibration Scheduled Date","First Calibration Scheduled date","First_Calibration_Scheduled_Date")
    cal_res = cal_resched_col or _pick(df_f, "Calibration Rescheduled Date","Calibration Rescheduled date","Calibration_Rescheduled_Date")
    cal_done = cal_done_col or _pick(df_f, "Calibration Booking Date","Calibration Booked Date","Calibration Booking date","Calibration_Done_Date","Calibration_Booked_Date")

    # Working copy with parsed dates (day-first tolerant)
    df = df_f.copy()
    # Canonicalize JetLearn Deal Source values to avoid singular/plural/case mismatches
    def _canon_source(s: str) -> str:
        if not isinstance(s, str):
            return "Unknown"
        x = s.strip().lower()
        # common aliases
        if x in {"referral", "referrals"}:
            return "Referrals"
        if x in {"event", "events"}:
            return "Events"
        if x in {"pm - search", "pm search", "paid marketing - search", "pm_search"}:
            return "PM - Search"
        if x in {"pm - social", "pm social", "paid marketing - social", "pm_social"}:
            return "PM - Social"
        if x in {"organic"}:
            return "Organic"
        return s.strip()

    if _source and _source in df.columns:
        df["__src_norm"] = df[_source].apply(_canon_source)
    else:
        df["__src_norm"] = "Unknown"

    for col in [create_col, pay_col, first_cal, cal_res, cal_done]:
        if col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, utc=False, infer_datetime_format=True)

    # ---------------- UI ----------------
    # Left: parameter list (labels only); Right: windows configuration is fixed (A..F)
    st.caption("Compare multiple **JetLearn Deal Source** windows side-by-side.")

    # Global windowing controls
    c0, c1, c2, c3 = st.columns([1,1,1,1])
    with c0:
        preset = st.radio("Date preset", ["Today","Yesterday","This Month","Custom"], index=2, key="lm_preset")
    today = date.today()
    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d, end_d = today - timedelta(days=1), today - timedelta(days=1)
    elif preset == "This Month":
        start_d, end_d = date(today.year, today.month, 1), today
    else:
        with c1:
            start_d = st.date_input("Start", value=date(today.year, today.month, 1), key="lm_start")
        with c2:
            end_d = st.date_input("End", value=today, key="lm_end")
        if start_d > end_d:
            st.warning("Start date is after end date. Please adjust."); return
    with c3:
        breakdown = st.selectbox("Breakdown for chart", ["Deal Source","Counsellor","Country"], index=0, key="lm_break")

    # Override dates for MTD mode
    if view_mode == "MTD":
        today = date.today()
        start_d = date(today.year, today.month, 1)
        end_d = today
    
    # Windows: fixed A..F with seeded sources
    all_sources = sorted(df["__src_norm"].dropna().astype(str).unique().tolist()) if "__src_norm" in df.columns else []
    def seed_or_all(name: str):
        return name if name in all_sources else "All"
    # Canonical mapping for windows and Remaining
    all_sources = sorted(df["__src_norm"].dropna().astype(str).unique().tolist()) if "__src_norm" in df.columns else []
    def present(name: str): return name if (name in all_sources) else None

    # Compute remaining dynamically (sources not in the named set that are present)
    named_set = {"Referrals","Events","Organic","PM - Search","PM - Social"}
    remaining_set = [s for s in all_sources if s not in named_set]

    windows = [
        ("A", "All"),
        ("B", present("Referrals")),
        ("C", present("Events")),
        ("D", present("Organic")),
        ("E", present("PM - Search")),
        ("F", present("PM - Social")),
        ("G", "__REMAINING__" if remaining_set else None),
    ]
    
    # Legend for windows
    try:
        import pandas as _pd
        legend_rows = []
        for lab, src in windows:
            if src == "__REMAINING__":
                label = "Remaining (others)"
            elif src is None:
                # show that a named source isn't present in this data
                name_map = {"B":"Referrals","C":"Events","D":"Organic","E":"PM - Search","F":"PM - Social"}
                label = f"{name_map.get(lab, 'All')} (not in data)"
            else:
                label = src
            legend_rows.append((lab, label))
        legend_df = _pd.DataFrame(legend_rows, columns=["Window","JetLearn Deal Source"])
        st.markdown("**Window legend**")
        st.dataframe(legend_df, use_container_width=True, hide_index=True)
    except Exception as _e:
        st.caption(f"Window legend unavailable: {_e}")

    # ---------------- Logic ----------------
    # Date masks PER METRIC using its own date column (but same global start/end)
    def _mask(df_in: pd.DataFrame, col: str | None):
        if not col or col not in df_in.columns:
            return pd.Series([False]*len(df_in), index=df_in.index)
        s = pd.to_datetime(df_in[col], errors="coerce")
        return (s.dt.date >= start_d) & (s.dt.date <= end_d)

    # Build a metrics dict per window
    rows = []
    per_window_tables = []

    
    for label, src in windows:
        dfw = df.copy()
        # Source filtering per window
        if "__src_norm" in dfw.columns:
            if src == "All" or src is None:
                pass
            elif src == "__REMAINING__":
                _used = {"Referrals","Events","Organic","PM - Search","PM - Social"}
                dfw = dfw[~dfw["__src_norm"].astype(str).isin(list(_used))]
            else:
                dfw = dfw[dfw["__src_norm"].astype(str) == src]

        # --- Metrics per mode ---
        if view_mode == "Cohort":
            # Cohort mode: cohort membership by Create Date in [start_d, end_d]
            cd = pd.to_datetime(dfw[create_col], errors="coerce", dayfirst=True)
            base = (cd.dt.date >= start_d) & (cd.dt.date <= end_d)

            leads = int(base.sum())
            # Presence-based counts within the cohort
            enrolls = int((base & dfw[pay_col].notna()) .sum()) if pay_col in dfw.columns else 0
            t_sched = int((base & dfw[first_cal].notna()).sum()) if first_cal and first_cal in dfw.columns else 0
            t_resch = int((base & dfw[cal_res].notna()).sum()) if cal_res and cal_res in dfw.columns else 0
            cal_done_ct = int((base & dfw[cal_done].notna()).sum()) if cal_done and cal_done in dfw.columns else 0
        else:
            # MTD/Date-range mode: use the parameter's own date column for the window
            def _mask_col(col):
                if not col or col not in dfw.columns: return pd.Series([False]*len(dfw), index=dfw.index)
                s = pd.to_datetime(dfw[col], errors="coerce", dayfirst=True)
                return (s.dt.date >= start_d) & (s.dt.date <= end_d)

            leads = int(_mask_col(create_col).sum())
            enrolls = int(_mask_col(pay_col).sum()) if pay_col in dfw.columns else 0
            t_sched = int(_mask_col(first_cal).sum()) if first_cal else 0
            t_resch = int(_mask_col(cal_res).sum()) if cal_res else 0
            cal_done_ct = int(_mask_col(cal_done).sum()) if cal_done else 0

        conv = round((enrolls / leads * 100.0), 1) if leads > 0 else 0.0

        rows.append({"Parameter": "Total leads", label: leads})
        rows.append({"Parameter": "Total enrollments", label: enrolls})
        rows.append({"Parameter": "Trial scheduled", label: t_sched})
        rows.append({"Parameter": "Trial rescheduled", label: t_resch})
        rows.append({"Parameter": "Calibration done", label: cal_done_ct})
        rows.append({"Parameter": "Lead → Enrollment %", label: conv})

        # For per-window breakdowns used in charts
        dfw = dfw.assign(__win=label)
        per_window_tables.append(dfw)
    # Merge window columns onto parameter rows
    if rows:
        param_df = pd.DataFrame(rows)
        param_df = param_df.groupby("Parameter").sum(numeric_only=True).reset_index()

        # UI layout: Left parameters list, Right metrics table
        lcol, rcol = st.columns([1,3])
        with lcol:
            st.markdown("**Parameters**")
            st.markdown("""- Total leads
- Total enrollments
- Trial scheduled
- Trial rescheduled
- Calibration done
- Lead → Enrollment %""")
        with rcol:
            st.markdown("**Windows (A..G)**")
            st.dataframe(param_df.set_index("Parameter"), use_container_width=True)
            st.download_button(
                "Download CSV — Lead mix metrics (A..G)",
                param_df.to_csv(index=False).encode("utf-8"),
                file_name="lead_mix_metrics.csv",
                mime="text/csv",
                key="lm_dl_metrics"
            )

    # ---------------- Chart ----------------

    df_all = pd.concat(per_window_tables, ignore_index=True) if per_window_tables else pd.DataFrame()
    if not df_all.empty:
        # Apply baseline mask by Create Date (for both modes)
        m_leads = (pd.to_datetime(df_all[create_col], errors="coerce", dayfirst=True).dt.date.between(start_d, end_d))
        plot_df = df_all.loc[m_leads].copy()

        if view_mode == "Cohort":
            # Cohort by Create Date (YYYY-MM), colored by Window
            plot_df["_cohort"] = pd.to_datetime(plot_df[create_col], errors="coerce", dayfirst=True).dt.to_period("M").astype(str)
            agg = plot_df.groupby(["_cohort", "__win"]).size().reset_index(name="count")
            base = alt.Chart(agg).encode(
                x=alt.X("_cohort:N", title="Create-Date Cohort (YYYY-MM)"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("__win:N", title="Window (A..G)"),
                tooltip=[alt.Tooltip("_cohort:N", title="Cohort"), alt.Tooltip("__win:N", title="Window"), "count:Q"]
            )
            st.altair_chart(base.mark_bar(), use_container_width=True)
        else:
            # MTD (or date-range) view: breakdown by selected dimension vs Window
            if breakdown == "Deal Source":
                bcol = "__src_norm"
            elif breakdown == "Counsellor":
                bcol = _counsellor
            else:
                bcol = _country

            if bcol and bcol in plot_df.columns:
                plot_df[bcol] = plot_df[bcol].fillna("Unknown").astype(str)
            else:
                bcol = "__win"  # fallback

            agg = plot_df.groupby(["__win", bcol]).size().reset_index(name="count")
            base = alt.Chart(agg).encode(
                x=alt.X("__win:N", title="Window (A..G)"),
                y=alt.Y("count:Q", title="Count (Create Date in range)"),
                color=alt.Color(f"{bcol}:N", title=breakdown if bcol != "__win" else "Window"),
                tooltip=["__win:N", alt.Tooltip(f"{bcol}:N", title=breakdown if bcol != "__win" else "Window"), "count:Q"]
            )
            st.altair_chart(base.mark_bar(), use_container_width=True)
    else:
        st.info("No data available for the current configuration.")
    




# ------------------------------
# Performance — Referral performance
# ------------------------------
def _render_performance_referral_performance(
    df_f,
    create_col: str | None = None,
    pay_col: str | None = None,
    ref_intent_col: str | None = None,
    country_col: str | None = None,
    counsellor_col: str | None = None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date, timedelta

    st.subheader("Performance — Referral performance")

    # Mode toggle (match Lead mix semantics: swap behavior under the same labels)
    view_mode = st.radio("Mode", ["MTD","Cohort"], index=0, horizontal=True, key="rp_mode")

    # --- Guards ---
    if create_col is None or create_col not in df_f.columns:
        st.error("Create Date column not found."); return
    if pay_col is None or pay_col not in df_f.columns:
        st.error("Payment Received Date column not found."); return

    # --- Column helpers ---
    def _pick(df: pd.DataFrame, *cands):
        for c in cands:
            if c and c in df.columns:
                return c
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if isinstance(c, str) and c.lower().strip() in low:
                return low[c.lower().strip()]
        return None

    _country = country_col or _pick(df_f, "Country")
    _counsellor = counsellor_col or _pick(df_f, "Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor")
    _ref = ref_intent_col or _pick(df_f, "Referral Intent Source", "Referral intent source", "Referral_Intent_Source", "Referral Intent", "Referral Source")

    # Working copy & parse dates
    df = df_f.copy()
    df[create_col] = pd.to_datetime(df[create_col], errors="coerce", dayfirst=True, utc=False, infer_datetime_format=True)
    df[pay_col] = pd.to_datetime(df[pay_col], errors="coerce", dayfirst=True, utc=False, infer_datetime_format=True)

    # Normalize referral intent source
    if _ref and _ref in df.columns:
        df["__ref"] = df[_ref].fillna("Unknown").astype(str).str.strip()
    else:
        df["__ref"] = "Unknown"

    # --- Optional filters ---
    col_f1, col_f2 = st.columns([1,1])
    with col_f1:
        hide_unknown = st.checkbox("Hide 'Unknown' intents", value=False, key="rp_hide_unknown")
    if hide_unknown:
        df = df[df["__ref"] != "Unknown"]

    st.caption("Mode help — MTD: per-metric date windows; Cohort: Create-Date cohort membership with presence-based enrollments.")

    # ---------------- Global controls ----------------
    c0, c1, c2, c3 = st.columns([1,1,1,1])
    with c0:
        preset = st.radio("Date preset", ["Today","Yesterday","This Month","Custom"], index=2, key="rp_preset")
    today = date.today()
    if preset == "Today":
        start_d = today; end_d = today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = today - timedelta(days=1)
    elif preset == "This Month":
        start_d = date(today.year, today.month, 1); end_d = today
    else:
        with c1:
            start_d = st.date_input("Start", value=date(today.year, today.month, 1), key="rp_start")
        with c2:
            end_d = st.date_input("End", value=today, key="rp_end")
        if start_d > end_d:
            st.warning("Start date is after end date."); return

    with c3:
        breakdown = st.selectbox("Breakdown for charts", ["Counsellor","Country"], index=0, key="rp_break")

    # MTD override (keep parity with lead-mix inversion semantics)
    if view_mode == "MTD":
        today = date.today()
        start_d = date(today.year, today.month, 1); end_d = today

    # Top-N
    topN = st.selectbox("Top-N Referral Intent Sources", ["Top 10","All"], index=0, key="rp_top")
    topN_val = 10 if topN == "Top 10" else None

    # ------------ Helper masks per column ------------
    def mask_by(col):
        if not col or col not in df.columns: return pd.Series([False]*len(df), index=df.index)
        s = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        return (s.dt.date >= start_d) & (s.dt.date <= end_d)

    # ------------ Build base subset per mode ------------
    if view_mode == "MTD":
        # Date-window per metric
        deals_mask = mask_by(create_col)
        enrol_mask = mask_by(pay_col)
        base_deals = df.loc[deals_mask].copy()
        base_enrol = df.loc[enrol_mask].copy()
        cohort_subset = None
    else:
        # Cohort membership by Create Date
        cohort_mask = mask_by(create_col)
        cohort_subset = df.loc[cohort_mask].copy()
        base_deals = cohort_subset.copy()
        base_enrol = cohort_subset.loc[cohort_subset[pay_col].notna()].copy()

    # ------------ KPIs ------------
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Deals (create)", f"{len(base_deals):,}")
    with k2: st.metric("Enrollments", f"{len(base_enrol):,}")
    with k3: st.metric("Active Referral Intent Sources", f"{base_deals['__ref'].nunique():,}")
    top_ref = base_deals['__ref'].value_counts().head(1)
    tr_label, tr_val = (top_ref.index[0], int(top_ref.iloc[0])) if not top_ref.empty else ("—", 0)
    with k4: st.metric("Top Referral Intent", tr_label, f"{tr_val:,}")

    # ------------ Distribution table ------------
    deals_count = base_deals.groupby("__ref").size().rename("Deals").reset_index()
    enrol_count = base_enrol.groupby("__ref").size().rename("Enrollments").reset_index()
    dist = pd.merge(deals_count, enrol_count, on="__ref", how="outer").fillna(0)
    dist["Total"] = dist["Deals"] + dist["Enrollments"]
    dist = dist.sort_values("Total", ascending=False)
    if topN_val:
        dist = dist.head(topN_val)
    total_deals = dist["Deals"].sum() if len(dist) else 0
    total_enrol = dist["Enrollments"].sum() if len(dist) else 0
    dist["Deals_%"] = (dist["Deals"]/total_deals*100).round(1) if total_deals>0 else 0
    dist["Enroll_%"] = (dist["Enrollments"]/total_enrol*100).round(1) if total_enrol>0 else 0

    st.markdown("### Distribution — Referral Intent Source (Deals & Enrollments)")
    st.dataframe(dist[["__ref","Deals","Deals_%","Enrollments","Enroll_%"]].rename(columns={"__ref":"Referral Intent Source"}), use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Referral distribution",
        dist.to_csv(index=False).encode("utf-8"),
        file_name="referral_performance_distribution.csv",
        mime="text/csv",
        key="rp_dl_dist"
    )

    # ------------ Combined graph (Deals bar + Enroll line) ------------
    st.markdown("### Combined — Deals & Enrollments by Referral Intent")
    plot = dist.copy()
    plot = plot.rename(columns={"__ref":"Referral Intent"})
    bars = alt.Chart(plot).mark_bar().encode(
        x=alt.X("Referral Intent:N", sort=plot["Referral Intent"].tolist()),
        y=alt.Y("Deals:Q"),
        tooltip=["Referral Intent","Deals","Enrollments"]
    )
    line = alt.Chart(plot).mark_line(point=True).encode(
        x="Referral Intent:N",
        y=alt.Y("Enrollments:Q"),
        tooltip=["Referral Intent","Deals","Enrollments"]
    )
    st.altair_chart(bars + line, use_container_width=True)

    # ------------ Breakdown view ------------
    st.markdown("### Breakdown")
    _bcol = _counsellor if breakdown == "Counsellor" else _country
    if _bcol and _bcol in df.columns:
        if view_mode == "MTD":
            base_for_break = df.loc[mask_by(create_col)].copy()
        else:
            base_for_break = cohort_subset.copy()
        base_for_break[_bcol] = base_for_break[_bcol].fillna("Unknown").astype(str)
        base_for_break["Referral Intent"] = base_for_break["__ref"]
        keep_intents = dist["Referral Intent"].tolist() if "Referral Intent" in dist.columns else dist["__ref"].tolist()
        base_for_break = base_for_break[base_for_break["Referral Intent"].isin(keep_intents)]
        metric = st.selectbox("Metric", ["Deals","Enrollments"], index=0, key="rp_metric")
        if metric == "Deals":
            if view_mode == "MTD":
                filt = mask_by(create_col).reindex(base_for_break.index, fill_value=False)
                dfm = base_for_break.loc[filt].copy()
            else:
                # Cohort mode: base_for_break is already cohort-limited by Create Date
                dfm = base_for_break.copy()
        else:
            if view_mode == "MTD":
                enrol_mask = mask_by(pay_col).reindex(base_for_break.index, fill_value=False)
                dfm = base_for_break.loc[enrol_mask].copy()
            else:
                # Cohort mode: presence-based within cohort
                dfm = base_for_break.loc[base_for_break[pay_col].notna()].copy()
        agg = dfm.groupby([_bcol,"Referral Intent"]).size().reset_index(name="count")
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X(f"{_bcol}:N", title=breakdown),
            y=alt.Y("count:Q", title=f"{metric} count"),
            color="Referral Intent:N",
            tooltip=[_bcol, "Referral Intent", "count"]
        )
        st.altair_chart(chart, use_container_width=True)
        st.download_button(
            "Download CSV — Breakdown",
            agg.to_csv(index=False).encode("utf-8"),
            file_name="referral_performance_breakdown.csv",
            mime="text/csv",
            key="rp_dl_break"
        )
    else:
        st.info(f"No column found for selected breakdown: {breakdown}")

    # ------------ Month-by-month comparison (trend) ------------
    st.markdown("### Month-by-month comparison (trend)")
    intents = dist.sort_values("Deals", ascending=False)["Referral Intent Source" if "Referral Intent Source" in dist.columns else "__ref"].head(3).tolist()
    multi = st.multiselect("Referral Intent to compare", intents, default=intents, key="rp_ints")
    def month_key(s): return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.to_period("M").astype(str)
    if view_mode == "MTD":
        df_deals_ts = df.loc[mask_by(create_col)].copy()
        df_enrl_ts = df.loc[mask_by(pay_col)].copy()
    else:
        df_deals_ts = cohort_subset.copy()
        df_enrl_ts = cohort_subset.loc[cohort_subset[pay_col].notna()].copy()
    df_deals_ts["month"] = month_key(df_deals_ts[create_col])
    df_enrl_ts["month"] = month_key(df_enrl_ts[pay_col])
    ts_deals = df_deals_ts.groupby(["__ref","month"]).size().reset_index(name="Deals")
    ts_enrl = df_enrl_ts.groupby(["__ref","month"]).size().reset_index(name="Enrollments")
    ts = pd.merge(ts_deals, ts_enrl, on=["__ref","month"], how="outer").fillna(0)
    ts = ts[ts["__ref"].isin(multi)]
    if ts.empty:
        st.info("No time-series data in selection.")
    else:
        ts_long = ts.melt(id_vars=["__ref","month"], value_vars=["Deals","Enrollments"], var_name="Metric", value_name="count")
        base = alt.Chart(ts_long).encode(
            x=alt.X("month:N", title="Month (YYYY-MM)"),
            y=alt.Y("count:Q", title="Count"),
            color="Metric:N",
            column=alt.Column("__ref:N", title="Referral Intent")
        )
        st.altair_chart(base.mark_line(point=True), use_container_width=True)
        st.download_button(
            "Download CSV — Month-by-month trend",
            ts.to_csv(index=False).encode("utf-8"),
            file_name="referral_performance_trend.csv",
            mime="text/csv",
            key="rp_dl_trend"
        )






# ------------------------------
# Performance — Slow Working Deals
# ------------------------------
def _render_performance_slow_working_deals(
    df_f,
    create_col: str | None = None,
    last_activity_col: str | None = None,
    country_col: str | None = None,
    counsellor_col: str | None = None,
    source_col: str | None = None,
    times_contacted_col: str | None = None,
    sales_activity_col: str | None = None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date, timedelta

    st.subheader("Performance — Slow Working Deals")

    def _pick(df: pd.DataFrame, *cands):
        for c in cands:
            if c and c in df.columns:
                return c
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if isinstance(c, str) and c.lower().strip() in low:
                return low[c.lower().strip()]
        return None

    _create = create_col or _pick(df_f, "Create Date","Created At","Create_Date","Created")
    _last = last_activity_col or _pick(df_f, "Last Activity Date","Last_Activity_Date","[Last] Activity Date","Last activity date")
    _country = country_col or _pick(df_f, "Country")
    _counsellor = counsellor_col or _pick(df_f, "Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor")
    _source = source_col or _pick(df_f, "JetLearn Deal Source","Deal Source","Source")
    _times = times_contacted_col or _pick(df_f, "Number of times contacted","Times Contacted","No. of times contacted","Times_Contacted","Contacted Count")
    _sales = sales_activity_col or _pick(df_f, "Number of sales activity","Sales Activity Count","Sales Activities","Sales_Activity_Count")

    if not _create:
        st.error("Create Date column not found."); return
    if not _last:
        st.error("Last Activity Date column not found."); return

    df = df_f.copy()
    df[_create] = pd.to_datetime(df[_create], errors="coerce", dayfirst=True, utc=False, infer_datetime_format=True)
    df[_last] = pd.to_datetime(df[_last], errors="coerce", dayfirst=True, utc=False, infer_datetime_format=True)
    for c in [_country, _counsellor, _source]:
        if c and c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    if _times and _times in df.columns:
        df["_times_contacted_num"] = pd.to_numeric(df[_times], errors="coerce")
    else:
        df["_times_contacted_num"] = np.nan
    if _sales and _sales in df.columns:
        df["_sales_activity_num"] = pd.to_numeric(df[_sales], errors="coerce")
    else:
        df["_sales_activity_num"] = np.nan

    c0,c1,c2,c3 = st.columns([1,1,1,1])
    with c0:
        view_mode = st.radio("Mode", ["MTD","Cohort"], index=0, horizontal=True, key="swd_mode")
    with c1:
        preset = st.radio("Date preset", ["Today","Yesterday","This Month","Custom"], index=2, key="swd_preset")
    today = date.today()
    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d, end_d = today - timedelta(days=1), today - timedelta(days=1)
    elif preset == "This Month":
        start_d, end_d = date(today.year, today.month, 1), today
    else:
        with c2:
            start_d = st.date_input("Start", value=date(today.year, today.month, 1), key="swd_start")
        with c3:
            end_d = st.date_input("End", value=today, key="swd_end")
        if start_d > end_d:
            st.warning("Start date is after end date."); return

    # --- Create-Date filter (applies to ALL calculations in this pill) ---
    cr1, cr2 = st.columns([1,1])
    with cr1:
        cd_start = st.date_input("Create Date — Start", value=date(today.year, today.month, 1), key="swd_cd_start")
    with cr2:
        cd_end = st.date_input("Create Date — End", value=end_d, key="swd_cd_end")
    if cd_start > cd_end:
        st.warning("Create Date start is after end. Please adjust."); return

    tcol1, tcol2, tcol3 = st.columns([1,1,2])
    with tcol1:
        idle_days = st.slider("Idle threshold (days since last activity ≥)", min_value=1, max_value=90, value=7, step=1, key="swd_idle")
    with tcol2:
        include_never = st.checkbox("Include never-contacted (no last activity)", value=True, key="swd_never")
    with tcol3:
        breakdown = st.selectbox("Breakdown", ["Counsellor","Country","Deal Source"], index=0, key="swd_break")
    stack_choice = st.checkbox("Stack by status (Never / Idle / Recent)", value=True, key="swd_stack")

    if view_mode == "Cohort":
        mask_cohort = (df[_create].dt.date >= start_d) & (df[_create].dt.date <= end_d)
        df = df.loc[mask_cohort].copy()

    # Always scope to deals created within the Create-Date range
    df = df.loc[(df[_create].dt.date >= cd_start) & (df[_create].dt.date <= cd_end)].copy()

    ref_ts = pd.Timestamp(end_d)
    days_since = (ref_ts - df[_last]).dt.days
    df["_days_since_last"] = days_since

    df["_status"] = "Recent"
    df.loc[(df[_last].isna()) & include_never, "_status"] = "Never"
    idle_cond = (df[_last].notna()) & (df["_days_since_last"] >= idle_days)
    df.loc[idle_cond, "_status"] = "Idle"
    if include_never:
        df.loc[df[_last].isna(), "_status"] = "Never"

    if include_never:
        df_idle = df[(df["_status"].isin(["Idle","Never"]))].copy()
    else:
        df_idle = df[(df["_status"]=="Idle")].copy()

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Idle deals", f"{len(df_idle):,}")
    with k2: st.metric("Never-contacted", f"{int((df_idle['_status']=='Never').sum()):,}")
    with k3:
        med_days = int(pd.to_numeric(df_idle["_days_since_last"], errors="coerce").median(skipna=True) or 0) if len(df_idle) else 0
        st.metric("Median days since last", f"{med_days}")
    with k4:
        st.metric("Avg #Times Contacted", f"{pd.to_numeric(df_idle['_times_contacted_num'], errors='coerce').mean(skipna=True):.1f}")
    with k5:
        st.metric("Avg #Sales Activity", f"{pd.to_numeric(df_idle['_sales_activity_num'], errors='coerce').mean(skipna=True):.1f}")

    dim = _counsellor if breakdown=="Counsellor" else (_country if breakdown=="Country" else _source)
    if not dim or dim not in df.columns:
        st.info(f"No column found for selected breakdown: {breakdown}")
        return

    grp_idle = (df_idle
                .assign(**{dim: df_idle[dim].fillna("Unknown").astype(str)})
                .groupby([dim,"_status"], dropna=False)
                .size().reset_index(name="count"))

    pivot_idle = grp_idle.pivot_table(index=dim, columns="_status", values="count", aggfunc="sum", fill_value=0)
    pivot_idle["Total Idle"] = pivot_idle.get("Idle",0) + pivot_idle.get("Never",0)
    avg_contacts = df_idle.groupby(dim)["_times_contacted_num"].mean().round(1)
    avg_sales = df_idle.groupby(dim)["_sales_activity_num"].mean().round(1)
    pivot_idle["Avg #Times Contacted"] = avg_contacts
    pivot_idle["Avg #Sales Activity"] = avg_sales
    pivot_idle = pivot_idle.fillna(0).reset_index()

    st.markdown("### Idle deals summary")
    st.dataframe(pivot_idle, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Idle summary",
        pivot_idle.to_csv(index=False).encode("utf-8"),
        file_name="slow_working_deals_summary.csv",
        mime="text/csv",
        key="swd_dl_summary"
    )

    st.markdown("### Idle deals by " + breakdown)
    agg = (df_idle
           .assign(**{dim: df_idle[dim].fillna("Unknown").astype(str)})
           .groupby([dim] + (["_status"] if stack_choice else []))
           .size().reset_index(name="count"))
    base = alt.Chart(agg).encode(
        x=alt.X(f"{dim}:N", title=breakdown),
        y=alt.Y("count:Q", title="Idle deals"),
        tooltip=[f"{dim}:N","count:Q"] + (["_status:N"] if stack_choice else [])
    )
    if stack_choice:
        chart = base.mark_bar().encode(color=alt.Color("_status:N", title="Status"))
    else:
        chart = base.mark_bar()
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Month-by-month trend")
    df_idle["_cohort"] = df_idle[_create].dt.to_period("M").astype(str)
    ts = df_idle.groupby(["_cohort"]).size().reset_index(name="Idle deals")
    st.altair_chart(alt.Chart(ts).mark_line(point=True).encode(x=alt.X("_cohort:N", title="Cohort (YYYY-MM)"), y="Idle deals:Q", tooltip=["_cohort","Idle deals"]), use_container_width=True)

    with st.expander("Show detailed idle deals"):
        cols_to_show = [_create, _last]
        for c in [dim, _country, _counsellor, _source]:
            if c and c not in cols_to_show: cols_to_show.append(c)
        for c in ["_days_since_last","_status"]:
            if c not in cols_to_show: cols_to_show.append(c)
        if _times: cols_to_show.append(_times)
        if _sales: cols_to_show.append(_sales)
        detail = df_idle[cols_to_show].copy()
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV — Idle deals detail",
            detail.to_csv(index=False).encode("utf-8"),
            file_name="slow_working_deals_detail.csv",
            mime="text/csv",
            key="swd_dl_detail"
        )


# --- Performance / Deal stage (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Deal stage":
        _render_performance_deal_stage(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Deal stage error: {_e}")
    except Exception:
        pass



# --- Performance / Activity concentration (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Activity concentration":
        _render_performance_activity_concentration(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Activity concentration error: {_e}")
    except Exception:
        pass



# --- Performance / Lead mix (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Lead mix":
        _render_performance_lead_mix(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            pay_col=pay_col if 'pay_col' in globals() else None,
            first_cal_sched_col=first_cal_sched_col if 'first_cal_sched_col' in globals() else None,
            cal_resched_col=cal_resched_col if 'cal_resched_col' in globals() else None,
            cal_done_col=cal_done_col if 'cal_done_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Lead mix error: {_e}")
    except Exception:
        pass



# --- Performance / Referral performance (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Referral performance":
        _render_performance_referral_performance(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            pay_col=pay_col if 'pay_col' in globals() else None,
            ref_intent_col=ref_intent_source_col if 'ref_intent_source_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Referral performance error: {_e}")
    except Exception:
        pass



# --- Performance / Slow Working Deals (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Slow Working Deals":
        _render_performance_slow_working_deals(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            last_activity_col=last_activity_col if 'last_activity_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
            times_contacted_col=times_contacted_col if 'times_contacted_col' in globals() else None,
            sales_activity_col=sales_activity_col if 'sales_activity_col' in globals() else None,
        )
except Exception as _e:
    st.error(f"Slow Working Deals failed: {_e}")


def _cohort_safe_dt(s, dayfirst=True):
    return _pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
def _choose_record_id_col(df):
    for c in ["Record ID","Deal ID","ID","HubSpot ID","record_id","deal_id"]:
        if c in df.columns:
            return c
    # create a temporary ID if not present
    if "_tmp_record_id" not in df.columns:
        df = df.assign(_tmp_record_id=_np.arange(1, len(df)+1))
    return "_tmp_record_id"
def _age_band_series(age_s):
    # Make bands: 5-7, 8-10, 11-13, 14-16, 17+ (flexible for kids range)
    bins = [-_np.inf, 7, 10, 13, 16, _np.inf]
    labels = ["≤7","8–10","11–13","14–16","17+"]
    try:
        age_num = _pd.to_numeric(age_s, errors="coerce")
        return _pd.Categorical(_pd.cut(age_num, bins=bins, labels=labels, right=True), categories=labels, ordered=True)
    except Exception:
        return _pd.Series(_np.nan, index=age_s.index)


# ====================== End: Cohort Performance ======================
def _render_performance_cohort_performance(df_f=None, df=None, **kwargs):
    df = df_f if df_f is not None else df
    st.subheader("Cohort Performance")
    if df is None or df.empty:
        st.info("No data available.")
        return

    df = df.copy()

    # Detect columns
    create_col = next((c for c in df.columns if c.lower().strip() in ["create date","created date","deal created date","create_date"]), None)
    pay_col    = next((c for c in df.columns if c.lower().strip() in ["payment received date","payment date","enrolment date","enrollment date","payment_received_date"]), None)
    country_col= next((c for c in df.columns if "country" in c.lower()), None)
    age_col    = next((c for c in df.columns if c.lower()=="age" or "age " in c.lower()), None)
    source_col = next((c for c in df.columns if "deal source" in c.lower()), None)
    ac_col     = next((c for c in df.columns if "academic" in c.lower() and "counsel" in c.lower()), None)
    rec_col    = _choose_record_id_col(df)

    # Parse dates
    if create_col: df[create_col] = _cohort_safe_dt(df[create_col], dayfirst=True)
    if pay_col:    df[pay_col]    = _cohort_safe_dt(df[pay_col], dayfirst=True)

    # Enrolled flag + TTC
    df["_enrolled"] = df[pay_col].notna() if pay_col else False
    if create_col and pay_col:
        df["_ttc_days"] = (df[pay_col] - df[create_col]).dt.days
    else:
        df["_ttc_days"] = _np.nan

    # Build derived dims
    if age_col:
        df["_age_band"] = _age_band_series(df[age_col])
    dims_all = []
    if country_col: dims_all.append(("Country", country_col))
    if age_col:     dims_all.append(("Age band", "_age_band"))
    if source_col:  dims_all.append(("Deal Source", source_col))
    if ac_col:      dims_all.append(("Academic Counselor", ac_col))

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        date_scope = c1.selectbox("Date scope (for historical rates)", ["MTD","Cohort","Custom Range"], index=0, key="cohort_perf_scope")
        min_support = c2.slider("Min historical deals per cohort", 5, 100, 15, 1, key="cohort_perf_min_support")
        dims_labels = [d[0] for d in dims_all] or ["(no dimensions available)"]
        default_sel = dims_labels if dims_labels[0] != "(no dimensions available)" else []
        chosen_labels = c3.multiselect("Cohort dimensions", dims_labels, default=default_sel, key="cohort_perf_dims")

        # Date controls
        today = _pd.Timestamp.today().normalize()
        start_default = today.replace(day=1)
        end_default = today
        if date_scope == "MTD":
            start_dt, end_dt = start_default, end_default
        elif date_scope == "Cohort":
            # cohort: use each deal's creation month onward; operationally we'll just not restrict history
            start_dt, end_dt = None, None
        else:
            d1, d2 = st.date_input("Custom date range", value=(start_default.date(), end_default.date()), key="cohort_perf_dates")
            start_dt = _pd.Timestamp(d1) if d1 else None
            end_dt = _pd.Timestamp(d2) if d2 else None

    # Historical window
    hist = df.copy()
    if create_col and start_dt is not None:
        hist = hist[(hist[create_col] >= start_dt)]
    if create_col and end_dt is not None:
        hist = hist[(hist[create_col] <= end_dt + _pd.Timedelta(days=1) - _pd.Timedelta(seconds=1))]

    # Limit to cohorts with selected dims
    if chosen_labels and dims_all:
        chosen_pairs = [d for d in dims_all if d[0] in chosen_labels]
        grp_cols = [d[1] for d in chosen_pairs]
    else:
        grp_cols = [c for _, c in dims_all]

    # Compute historical conversion stats
    if grp_cols:
        grp = hist.groupby(grp_cols, dropna=False)
        hist_stats = grp.agg(total_deals=(_pd.NamedAgg(column=rec_col, aggfunc="count")),
                             enrolls=(_pd.NamedAgg(column="_enrolled", aggfunc="sum")),
                             avg_ttc_days=(_pd.NamedAgg(column="_ttc_days", aggfunc="mean"))).reset_index()
    else:
        hist_stats = _pd.DataFrame([{
            "total_deals": hist.shape[0],
            "enrolls": int(hist["_enrolled"].sum()),
            "avg_ttc_days": hist["_ttc_days"].mean()
        }])

    if not hist_stats.empty:
        hist_stats["probability"] = _np.where(hist_stats["total_deals"]>0, hist_stats["enrolls"]/hist_stats["total_deals"], _np.nan)
    else:
        st.warning("No historical data to compute probabilities.")
        return

    # Apply min support
    hist_stats = hist_stats[(hist_stats["total_deals"] >= min_support) & (hist_stats["probability"].notna())]

    # Open deals and expected conversions
    open_mask = ~df["_enrolled"].astype(bool)
    open_df = df.loc[open_mask].copy()
    if grp_cols:
        open_grp = open_df.groupby(grp_cols, dropna=False).agg(open_deals=(_pd.NamedAgg(column=rec_col, aggfunc="count"))).reset_index()
    else:
        open_grp = _pd.DataFrame([{"open_deals": open_df.shape[0]}])

    merged = _pd.merge(open_grp, hist_stats, how="inner", on=grp_cols if grp_cols else None)
    if merged.empty:
        st.info("No cohorts meet the support threshold or no open deals in those cohorts.")
        return

    merged["expected_conversions"] = merged["open_deals"] * merged["probability"]

    # Likely convert-by date (average across open deals per cohort)
    if create_col:
        # For each open deal, est_date = create + avg_ttc_days(cohort)
        if not grp_cols:
            # single cohort case
            open_df["_est_dt"] = open_df[create_col] + _pd.to_timedelta(merged["avg_ttc_days"].iloc[0], unit="D")
            likely_avg = open_df["_est_dt"].mean()
            merged["likely_convert_by"] = likely_avg
        else:
            # Map avg_ttc_days to each open deal by cohort keys
            key_cols = grp_cols
            hist_key = merged[key_cols + ["avg_ttc_days"]].drop_duplicates()
            open_w = _pd.merge(open_df, hist_key, on=key_cols, how="left")
            open_w["_est_dt"] = open_w[create_col] + _pd.to_timedelta(open_w["avg_ttc_days"], unit="D")
            # Compute average per cohort
            est_avg = open_w.groupby(key_cols, dropna=False)["_est_dt"].apply(lambda s: _pd.to_datetime(s, errors="coerce").mean()).reset_index().rename(columns={"_est_dt":"likely_convert_by"})
            merged = _pd.merge(merged, est_avg, on=key_cols, how="left")
    else:
        merged["likely_convert_by"] = _pd.NaT

    # Build display columns
    def _fmt_cohort_row(row):
        parts = []
        for lbl, col in (dims_all):
            if grp_cols and col not in grp_cols: 
                continue
            val = row.get(col, _np.nan)
            if _pd.isna(val):
                continue
            parts.append(f"{lbl}={val}")
        return " · ".join(parts) if parts else "(All deals)"

    merged["Cohort"] = merged.apply(_fmt_cohort_row, axis=1)
    merged["Probability (%)"] = (merged["probability"]*100).round(1)
    merged["Avg TTC (days)"] = merged["avg_ttc_days"].round(1)
    merged["Likely Convert-By"] = _pd.to_datetime(merged["likely_convert_by"]).dt.date

    # Prepare Record IDs list per cohort
    if grp_cols:
        # slice open_df by cohort and gather IDs
        keys = grp_cols
        ids_per = open_df.groupby(keys, dropna=False)[rec_col].apply(lambda s: ", ".join(map(str, s.tolist()))).reset_index().rename(columns={rec_col:"Record IDs"})
        merged = _pd.merge(merged, ids_per, on=keys, how="left")
    else:
        merged["Record IDs"] = ", ".join(map(str, open_df[rec_col].tolist()))

    # Rank Top-4
    merged = merged.sort_values(["expected_conversions","probability","open_deals"], ascending=[False, False, False]).reset_index(drop=True)
    merged.insert(0, "Rank", _np.arange(1, len(merged)+1))
    top4 = merged.head(4).copy()

    # Display chart
    chart = _alt.Chart(top4).mark_bar().encode(
        x=_alt.X("Cohort:N", sort="-y", title="Cohort"),
        y=_alt.Y("expected_conversions:Q", title="Expected Conversions"),
        tooltip=["Rank","Cohort","open_deals","Probability (%)","expected_conversions","Avg TTC (days)","Likely Convert-By"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    # Display table
    display_cols = ["Rank","Cohort","open_deals","Probability (%)","expected_conversions","Avg TTC (days)","Likely Convert-By","Record IDs"]
    nice = top4.rename(columns={
        "open_deals":"Open Deals",
        "expected_conversions":"Expected Conversions"
    })[display_cols]
    st.dataframe(nice, use_container_width=True)

    # Downloads
    cdl1, cdl2 = st.columns([1,1])
    csv_bytes = nice.to_csv(index=False).encode("utf-8")
    cdl1.download_button("Download CSV", data=csv_bytes, file_name="cohort_performance_top4.csv", mime="text/csv", key="dl_cohort_csv")
    try:
        import io
        import pandas as _pd2
        buffer = io.BytesIO()
        with _pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            nice.to_excel(writer, index=False, sheet_name="Top4 Cohorts")
        cdl2.download_button("Download Excel", data=buffer.getvalue(), file_name="cohort_performance_top4.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_cohort_xlsx")
    except Exception:
        pass

# --- Performance / Cohort performance (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Cohort performance":
        _render_performance_cohort_performance(
            df_f=df_f
        )
except Exception as _e:
    st.error(f"Cohort performance failed: {_e}")
    try:
        st = __import__("streamlit")
        st.error(f"Slow Working Deals error: {_e}")
    except Exception:
        pass


# --- Performance / Original source (explicit master gate) ---
try:
    if 'master' in globals() and master == "Performance" and 'view' in globals() and view == "Original source":
        _render_performance_original_source(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            enrol_col=pay_col if 'pay_col' in globals() else None,
            drill1_col="Original Traffic Source Drill-Down 1",
            drill2_col="Original Traffic Source Drill-Down 2",
        )
except Exception as _e:
    try: st.warning(f"Original source failed: {_e}")
    except Exception: pass





if master == "Performance" and view == "Quick View":
    _render_performance_quick_view(
        df_f,
        create_col=create_col,
        pay_col=pay_col,
        first_cal_sched_col=first_cal_sched_col,
        cal_resched_col=cal_resched_col,
        cal_done_col=cal_done_col,
        source_col=source_col,
    )





# ======================
# Performance ▶ Pipeline (Unified Metric, Global Booking Filter, Graph Styles)
# ======================
def _render_performance_pipeline(
    df_f: pd.DataFrame,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    cal_done_col: str | None,
    slot_col: str | None,
    counsellor_col: str | None,
    country_col: str | None,
    source_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta
    import altair as alt

    st.subheader("Performance — Pipeline")
    st.caption("Choose a metric and visualize counts day-wise. Booking filter applies to all metrics.")

    # Date presets
    preset = st.selectbox("Date range", ["Today", "Yesterday", "This Month", "Custom"], index=0, key="pipe_range_preset")
    if preset == "Today":
        date_start = date.today(); date_end = date.today()
    elif preset == "Yesterday":
        date_start = date.today() - timedelta(days=1); date_end = date_start
    elif preset == "This Month":
        _t = date.today(); date_start = _t.replace(day=1); date_end = _t
    else:
        c1, c2 = st.columns(2)
        with c1: date_start = st.date_input("From", value=date.today(), key="pipe_from")
        with c2: date_end = st.date_input("To", value=date.today(), key="pipe_to")
        if date_end < date_start:
            st.warning("End date is before start date; swapping automatically.")
            date_start, date_end = date_end, date_start

    # Metric choices
    metric = st.selectbox(
        "Metric",
        ["First Calibration Scheduled Date", "Calibration Rescheduled Date", "Calibration Booking Date", "Closed Lost Trigger Date"],
        index=0, key="pipe_metric"
    )

    # Graph style & breakdown
    graph_style = st.selectbox("Graph style", ["Stacked Bar", "Bar", "Line"], index=0, key="pipe_graph_style")
    dim = st.selectbox("Breakdown", ["Overall","Academic Counsellor","Country","JetLearn Deal Source"], index=0, key="pipe_dim")

    # Booking filter (applies to ALL metrics if slot column exists)
    booking_filter = st.radio(
        "Booking filter",
        ["All", "Pre-book (has Calibration Slot)", "Sales-book (no prebook)"],
        index=0, horizontal=True, key="pipe_booking_filter"
    )

    # helpers
    def pick(df, preferred, cands):
        if preferred and preferred in df.columns: return preferred
        for c in cands:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None

    owner_col  = pick(df_f, counsellor_col, ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor"])
    country_c  = pick(df_f, country_col, ["Country","Country Name"])
    source_c   = pick(df_f, source_col, ["JetLearn Deal Source","Deal Source","Source"])
    # Resolve slot column robustly
    if (not slot_col) or (slot_col not in df_f.columns):
        _slot_guess = None
        for c in df_f.columns:
            name = str(c).strip().lower()
            if 'calibration' in name and 'slot' in name and 'deal' in name:
                _slot_guess = c
                break
        slot_col = _slot_guess if _slot_guess else slot_col


    d = df_f.copy()

    def _to_dt(s):
        try:
            return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception:
            return pd.to_datetime(pd.Series([None]*len(s)), errors="coerce")

    # metric branches
    if metric == "First Calibration Scheduled Date":
        key = first_cal_sched_col if (first_cal_sched_col and first_cal_sched_col in d.columns) else None
        if not key:
            st.warning("First Calibration Scheduled Date column not found."); return
        d["_pipe_date"] = _to_dt(d[key])

    elif metric == "Calibration Rescheduled Date":
        key = cal_resched_col if (cal_resched_col and cal_resched_col in d.columns) else None
        if not key:
            st.warning("Calibration Rescheduled Date column not found."); return
        d["_pipe_date"] = _to_dt(d[key])

    elif metric == "Calibration Booking Date":
        # explicit or derive from slot
        explicit_booking_col = None
        for cand in ["Calibration Booking Date", "Cal Booking Date", "Booking Date (Calibration)"]:
            if cand in d.columns:
                explicit_booking_col = cand; break
        if explicit_booking_col:
            d["_pipe_date"] = _to_dt(d[explicit_booking_col])
        else:
            if not (slot_col and slot_col in d.columns):
                st.warning("No booking date column or slot column available to derive booking date."); return
            def extract_date(txt: str):
                if not isinstance(txt, str) or txt.strip() == "":
                    return None
                core = txt.split("-")[0]
                try: return pd.to_datetime(core, errors="coerce")
                except Exception: return None
            d["_pipe_date"] = d[slot_col].astype(str).map(extract_date)

    else:  # Closed Lost Trigger Date
        key = "Closed Lost Trigger Date"
        if key not in d.columns:
            st.warning("Column 'Closed Lost Trigger Date' not found. Please add it to use this metric."); return
        d["_pipe_date"] = _to_dt(d[key])

    # GLOBAL booking filter (applies to all metrics when slot is available)
    def _has_slot_col(series):
        s = series.astype(str).str.strip()
        return series.notna() & s.ne("") & ~s.str.lower().isin(["nan","none"])
    if slot_col and slot_col in d.columns:
        try:
            mask_has = _has_slot_col(d[slot_col])
            if booking_filter == "Pre-book (has Calibration Slot)":
                d = d.loc[mask_has].copy()
            elif booking_filter == "Sales-book (no prebook)":
                d = d.loc[~mask_has].copy()
        except Exception:
            pass

    # range filter
    d["_pipe_date_only"] = d["_pipe_date"].dt.date
    d = d.loc[(d["_pipe_date_only"] >= date_start) & (d["_pipe_date_only"] <= date_end)].copy()
    if d.empty:

        st.info("No records in the selected range / metric."); return

    # aggregation
    if dim == "Academic Counsellor" and not owner_col: dim = "Overall"
    if dim == "Country" and not country_c: dim = "Overall"
    if dim == "JetLearn Deal Source" and not source_c: dim = "Overall"

    if dim == "Overall":
        group_cols = ["_pipe_date_only"]
    elif dim == "Academic Counsellor":
        group_cols = ["_pipe_date_only", owner_col]
    elif dim == "Country":
        group_cols = ["_pipe_date_only", country_c]
    else:
        group_cols = ["_pipe_date_only", source_c]

    counts = d.groupby(group_cols, dropna=False).size().rename("Trials").reset_index()

    # fill missing days
    all_days = pd.date_range(date_start, date_end, freq="D").date
    if dim == "Overall":
        base_df = pd.DataFrame({"_pipe_date_only": all_days})
        counts = base_df.merge(counts, on="_pipe_date_only", how="left").fillna({"Trials":0})
    else:
        key_col = group_cols[1]
        rows = []
        for k, sub in counts.groupby(key_col, dropna=False):
            sub = sub.set_index("_pipe_date_only").reindex(all_days, fill_value=0).rename_axis("_pipe_date_only").reset_index()
            sub[key_col] = k
            rows.append(sub)
        counts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["_pipe_date_only", key_col, "Trials"])

    # KPIs
    today = date.today()
    today_cnt = int(counts.loc[counts["_pipe_date_only"] == today, "Trials"].sum())
    next7 = int(counts.loc[(counts["_pipe_date_only"] >= date_start) & (counts["_pipe_date_only"] <= min(date_end, date_start + timedelta(days=6))), "Trials"].sum())
    next30 = int(counts.loc[(counts["_pipe_date_only"] >= date_start) & (counts["_pipe_date_only"] <= min(date_end, date_start + timedelta(days=29))), "Trials"].sum())

    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Today", today_cnt)
    with k2: st.metric("Next 7 days (in range)", next7)
    with k3: st.metric("Next 30 days (in range)", next30)

    # Chart
    title = f"{metric} — Day-wise"
    if dim == "Overall":
        base = alt.Chart(counts)
        if graph_style == "Line":
            ch = base.mark_line().encode(
                x=alt.X("_pipe_date_only:T", title=None),
                y=alt.Y("Trials:Q", title="Count"),
                tooltip=[alt.Tooltip("_pipe_date_only:T", title="Date"), alt.Tooltip("Trials:Q")]
            ).properties(height=320, title=title)
        else:
            ch = base.mark_bar().encode(
                x=alt.X("_pipe_date_only:T", title=None),
                y=alt.Y("Trials:Q", title="Count"),
                tooltip=[alt.Tooltip("_pipe_date_only:T", title="Date"), alt.Tooltip("Trials:Q")]
            ).properties(height=320, title=title)
    else:
        key_col = group_cols[1]
        base = alt.Chart(counts)
        if graph_style == "Line":
            ch = base.mark_line().encode(
                x=alt.X("_pipe_date_only:T", title=None),
                y=alt.Y("Trials:Q", title="Count"),
                color=alt.Color(f"{key_col}:N", legend=alt.Legend(title=dim)),
                tooltip=[alt.Tooltip("_pipe_date_only:T", title="Date"), alt.Tooltip("Trials:Q"), alt.Tooltip(f"{key_col}:N", title=dim)]
            ).properties(height=360, title=title)
        else:
            stacking = "zero" if graph_style == "Stacked Bar" else None
            ch = base.mark_bar().encode(
                x=alt.X("_pipe_date_only:T", title=None),
                y=alt.Y("Trials:Q", title="Count", stack=stacking),
                color=alt.Color(f"{key_col}:N", legend=alt.Legend(title=dim)),
                tooltip=[alt.Tooltip("_pipe_date_only:T", title="Date"), alt.Tooltip("Trials:Q"), alt.Tooltip(f"{key_col}:N", title=dim)]
            ).properties(height=360, title=title)

    st.altair_chart(ch, use_container_width=True)

    st.download_button(
        "Download CSV — Pipeline",
        counts.rename(columns={"_pipe_date_only":"Date"}).to_csv(index=False).encode("utf-8"),
        file_name="pipeline_counts.csv",
        mime="text/csv"
    )

# ---- Dispatch for Pipeline ----
try:
    _cur_master = st.session_state.get('nav_master', '')
    _cur_view = st.session_state.get('nav_sub', '')
    if _cur_master == "Performance" and _cur_view == "Pipeline":
        _render_performance_pipeline(
            df_f=df_f,
            first_cal_sched_col=first_cal_sched_col,
            cal_resched_col=cal_resched_col,
            cal_done_col=cal_done_col,
            slot_col=calibration_slot_col if 'calibration_slot_col' in globals() else None,
            counsellor_col=counsellor_col if 'counsellor_col' in globals() else None,
            country_col=country_col if 'country_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
        )
except Exception as _e:
    try:
        st = __import__("streamlit")
        st.error(f"Pipeline view error: {_e}")
    except Exception:
        pass



# ======================
# Performance ▶ Sales Activity
# ======================



# ======================
# Added: Original Source drill-down filters (keeps everything else intact)
# ======================
def _render_original_source_drill_filters(df_base, create_col, pay_col):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date

    # Resolve the two drilldown columns robustly
    def _find(df, cands):
        for c in cands:
            if c and c in df.columns:
                return c
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if isinstance(c, str) and c.lower().strip() in low:
                return low[c.lower().strip()]
        return None

    d1 = _find(df_base, [
        "Original Traffic Source Drill-Down 1",
        "Original Traffic Source Drilldown 1",
        "Original Traffic Source Drill Down 1",
        "Original Traffic Source Drill-Down One",
        "Original traffic source drill-down 1",
        "Original Traffic Drill-Down 1",
    ])
    d2 = _find(df_base, [
        "Original Traffic Source Drill-Down 2",
        "Original Traffic Source Drilldown 2",
        "Original Traffic Source Drill Down 2",
        "Original traffic source drill-down 2",
        "Original Traffic Drill-Down 2",
    ])

    st.subheader("Performance — Original source")
    if not d1 and not d2:
        st.info("No 'Original Traffic Source Drill-Down 1/2' columns were found in your data. "
                "Columns expected: “Original Traffic Source Drill-Down 1/2”.")
        return

    # Build selectors
    c1, c2 = st.columns(2)
    df_local = df_base.copy()

    if d1:
        opts1 = ["All"] + sorted(df_local[d1].dropna().astype(str).unique().tolist())
        sel1 = c1.multiselect("Original Traffic Source Drill-Down 1", options=opts1, default=["All"], key="orig_src_dd1")
        if sel1 and "All" not in sel1:
            df_local = df_local[df_local[d1].astype(str).isin(sel1)]
    if d2:
        opts2 = ["All"] + sorted(df_local[d2].dropna().astype(str).unique().tolist())
        sel2 = c2.multiselect("Original Traffic Source Drill-Down 2", options=opts2, default=["All"], key="orig_src_dd2")
        if sel2 and "All" not in sel2:
            df_local = df_local[df_local[d2].astype(str).isin(sel2)]

    # Safety: coerce dates
    def _to_date(s):
        try:
            return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True).dt.date
        except Exception:
            return pd.to_datetime(pd.Series([None]*len(s)), errors="coerce").dt.date

    _create = _to_date(df_local[create_col]) if create_col in df_local.columns else None
    _pay    = _to_date(df_local[pay_col]) if pay_col in df_local.columns else None

    # Date filter controls (optional, to stay consistent with Performance tab patterns)
    today = date.today()
    with st.expander("Optional: Date window", expanded=False):
        mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="orig_src_mode")
        start, end = st.date_input("Payments / Activity window", value=(today.replace(day=1), today), key="orig_src_date")
        if isinstance(start, (list, tuple)):
            start, end = start
        if end < start:
            start, end = end, start

    # Apply window masks
    dfw = df_local.copy()
    if _create is not None and _pay is not None:
        c_in = (_create >= start) & (_create <= end)
        p_in = (_pay >= start) & (_pay <= end)
        if mode == "MTD":
            mask = p_in & c_in
        else:
            mask = p_in
        dfw = dfw.loc[mask].copy()

    # Aggregate summary
    group_keys = [c for c in [d1, d2] if c]
    if not group_keys:
        group_keys = ["(All)"]
        dfw = dfw.assign(**{"(All)": "All"})

    # Build metrics: Deals Created (Create Date count), Enrolments (Payment Received count)
    deals = _create if _create is not None else None
    pays  = _pay if _pay is not None else None

    df_counts = dfw.copy()
    # Prepare helper cols for grouping by dates
    if deals is not None:
        df_counts["_create_date"] = deals
    if pays is not None:
        df_counts["_pay_date"] = pays

    # Group
    g = df_counts.groupby(group_keys, dropna=False)
    summary = g.size().rename("Rows").to_frame()

    if deals is not None:
        summary = summary.join(g["_create_date"].apply(lambda s: int(s.notna().sum())).rename("Deals Created"))
    if pays is not None:
        summary = summary.join(g["_pay_date"].apply(lambda s: int(s.notna().sum())).rename("Enrolments"))

    summary = summary.reset_index().fillna("Unknown")

    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Original source (with drilldown filters)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="original_source_drilldown_summary.csv",
        mime="text/csv",
        key="dl_original_source"
    )

# Hook: render this block when user is on Performance ▶ Original source
try:
    if 'master' in globals() and 'view' in globals():
        _ms = master if 'master' in globals() else st.session_state.get('nav_master','')
        _vw = view if 'view' in globals() else st.session_state.get('nav_sub','')
        if _ms == "Performance" and _vw == "Original source":
            # Use the already-filtered df_f from earlier in the app, if available; fallback to df
            try:
                _df_base = df_f.copy()
            except Exception:
                _df_base = df.copy() if 'df' in globals() else None
            if _df_base is not None:
                # Reuse detected columns from earlier mapping
                _create_col = create_col if 'create_col' in globals() else "Create Date"
                _pay_col    = pay_col if 'pay_col' in globals() else "Payment Received Date"
                _render_original_source_drill_filters(_df_base, _create_col, _pay_col)
except Exception as _e:
    import streamlit as st
    st.warning(f"Original source drill-down filters could not be rendered: {type(_e).__name__}")



# ======================
# NEW: Performance ▶ Referral / No-Referral (Window A vs Window B)
# ======================
def _render_performance_referral_no_referral(
    df_f,
    create_col: str | None = None,
    pay_col: str | None = None,
    first_cal_sched_col: str | None = None,
    cal_resched_col: str | None = None,
    cal_done_col: str | None = None,
    source_col: str | None = None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date, timedelta
    from calendar import monthrange

    st.subheader("Performance — Referral / No-Referral (A vs B)")

    # --- Guards ---
    if create_col is None or create_col not in df_f.columns:
        st.error("Create Date column not found."); return
    if pay_col is None or pay_col not in df_f.columns:
        st.error("Payment Received Date column not found."); return

    # --- Column helpers ---
    def _pick(df: pd.DataFrame, *cands):
        for c in cands:
            if c and c in df.columns:
                return c
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if isinstance(c, str) and c.lower().strip() in low:
                return low[c.lower().strip()]
        return None

    _first = _pick(df_f, first_cal_sched_col, "First Calibration Scheduled Date", "First_Calibration_Scheduled_Date")
    _resch = _pick(df_f, cal_resched_col, "Calibration Rescheduled Date", "Calibration_Rescheduled_Date")
    _done  = _pick(df_f, cal_done_col,  "Calibration Done Date", "Calibration_Done_Date")
    _src   = _pick(df_f, source_col, "JetLearn Deal Source", "Deal Source", "Source")
    _refi  = _pick(df_f, "Referral Intent Source", "Referral_Intent_Source")

    def _dt(s: pd.Series):
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)

    d = df_f.copy()
    d["_create"] = _dt(d[create_col])
    d["_pay"]    = _dt(d[pay_col])
    d["_first"]  = _dt(d[_first]) if _first else pd.NaT
    d["_resch"]  = _dt(d[_resch]) if _resch else pd.NaT
    d["_done"]   = _dt(d[_done])  if _done  else pd.NaT
    d["_src"]    = d[_src].fillna("Unknown").astype(str).str.strip() if _src else "Unknown"
    d["_refi"]   = d[_refi].fillna("Unknown").astype(str).str.strip() if _refi else "Unknown"

    def _bounds_this_month(today: date):
        return date(today.year, today.month, 1), date(today.year, today.month, monthrange(today.year, today.month)[1])

    today = date.today()

    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="refnr_mode")
    cA, cB = st.columns(2)

    with cA:
        st.write("### Window A — Referral")
        presetA = st.radio("Date preset (A)", ["Today","Yesterday","This Month","Custom"], index=2, horizontal=True, key="refnr_preset_a")
        if presetA == "Today":
            a_start, a_end = today, today
        elif presetA == "Yesterday":
            a_end = today - timedelta(days=1); a_start = a_end
        elif presetA == "This Month":
            a_start, a_end = _bounds_this_month(today)
        else:
            c1, c2 = st.columns(2)
            with c1: a_start = st.date_input("Start (A)", value=_bounds_this_month(today)[0], key="refnr_start_a")
            with c2: a_end   = st.date_input("End (A)",   value=today, key="refnr_end_a")
            if a_end < a_start: a_start, a_end = a_end, a_start

        st.caption("Deal Source (A)")
        src_vals = sorted(d["_src"].dropna().unique().tolist())
        ref_defaults = [s for s in src_vals if "referr" in s.lower()]
        pickA = st.multiselect("Include sources (A)", options=src_vals, default=ref_defaults, key="refnr_src_a")

    with cB:
        st.write("### Window B — Non-Referral")
        presetB = st.radio("Date preset (B)", ["Today","Yesterday","This Month","Custom"], index=2, horizontal=True, key="refnr_preset_b")
        if presetB == "Today":
            b_start, b_end = today, today
        elif presetB == "Yesterday":
            b_end = today - timedelta(days=1); b_start = b_end
        elif presetB == "This Month":
            b_start, b_end = _bounds_this_month(today)
        else:
            c1, c2 = st.columns(2)
            with c1: b_start = st.date_input("Start (B)", value=_bounds_this_month(today)[0], key="refnr_start_b")
            with c2: b_end   = st.date_input("End (B)",   value=today, key="refnr_end_b")
            if b_end < b_start: b_start, b_end = b_end, b_start

        st.caption("Deal Source (B)")
        src_vals = sorted(d["_src"].dropna().unique().tolist())
        nonref_defaults = [s for s in src_vals if "referr" not in s.lower()]
        pickB = st.multiselect("Include sources (B)", options=src_vals, default=nonref_defaults, key="refnr_src_b")

    def _mask_window(d0, start_d, end_d, *, mode: str):
        m_create = d0["_create"].dt.date.between(start_d, end_d)
        m_pay    = d0["_pay"].dt.date.between(start_d, end_d)
        m_first  = d0["_first"].dt.date.between(start_d, end_d) if "_first" in d0 else (m_create & False)
        m_resch  = d0["_resch"].dt.date.between(start_d, end_d) if "_resch" in d0 else (m_create & False)
        m_done   = d0["_done"].dt.date.between(start_d, end_d)  if "_done"  in d0 else (m_create & False)
        if mode == "MTD":
            enrol = (m_pay & m_create)
            first = (m_first & m_create)
            resch = (m_resch & m_create)
            done  = (m_done  & m_create)
        else:
            enrol, first, resch, done = m_pay, m_first, m_resch, m_done
        return m_create, enrol, first, resch, done

    def _agg_side(d0, start_d, end_d, src_pick):
        dd = d0[d0["_src"].isin(src_pick)].copy()
        if dd.empty:
            return dict(Deals=0, Enrolments=0, FirstCal=0, CalResch=0, CalDone=0,
                        ReferralGenerated=0, SelfGenReferrals=0, L2E_pct=np.nan)
        m_create, m_enrol, m_first, m_resch, m_done = _mask_window(dd, start_d, end_d, mode=mode)
        deals = int(m_create.sum())
        enrol = int(m_enrol.sum())
        first = int(m_first.sum())
        resch = int(m_resch.sum())
        done  = int(m_done.sum())
        # Referral generated = Deal Source contains 'referr' (case-insensitive), created in window
        ref_gen = int((m_create & dd["_src"].astype(str).str.contains("referr", case=False, na=False)).sum())
        # Self-generated referrals = Referral Intent Source == 'Sales Generated' (robust), created in window
        if "_refi" in dd:
            rif = dd["_refi"].astype(str).str.strip()
            rif_norm = rif.str.replace(r"[\s\-_]+", "", regex=True).str.casefold()
            exact_norm = (rif_norm == "salesgenerated")
            token_match = rif.str.contains(r"(?i)\bsales\b.*\bgenerated\b")
            match_sales_gen = exact_norm | token_match
            self_gen = int((m_create & match_sales_gen).sum())
        else:
            self_gen = 0
        l2e = (enrol / deals * 100.0) if deals else np.nan
        return dict(Deals=deals, Enrolments=enrol, FirstCal=first, CalResch=resch, CalDone=done,
                    ReferralGenerated=ref_gen, SelfGenReferrals=self_gen, L2E_pct=l2e)

    # ---- Compute & Render ----
    try:
        resA = _agg_side(d, a_start, a_end, pickA or [])
        resB = _agg_side(d, b_start, b_end, pickB or [])

        def _kpi(title, a, b, suffix=""):
            c1, c2, c3 = st.columns([2,1,1])
            with c1: st.markdown(f"**{title}**")
            with c2: st.metric("A (Referral)", f"{a:.0f}{suffix}" if isinstance(a,(int,float)) and not np.isnan(a) else "—")
            with c3: st.metric("B (Non-Referral)", f"{b:.0f}{suffix}" if isinstance(b,(int,float)) and not np.isnan(b) else "—")

        st.divider()
        _kpi("Deals created", resA["Deals"], resB["Deals"])
        _kpi("Enrollments", resA["Enrolments"], resB["Enrolments"])
        _kpi("Trial Scheduled", resA["FirstCal"], resB["FirstCal"])
        _kpi("Trial Rescheduled", resA["CalResch"], resB["CalResch"])
        _kpi("Calibration Done", resA["CalDone"], resB["CalDone"])
        _kpi("Referral generated (JetLearn Deal Source = referrals)", resA["ReferralGenerated"], resB["ReferralGenerated"])
        _kpi("Self-generated referrals (Referral Intent Source = Sales Generated)", resA["SelfGenReferrals"], resB["SelfGenReferrals"])
        _kpi("Lead → Enrollment %", resA["L2E_pct"], resB["L2E_pct"], suffix="%")

        chart_df = pd.DataFrame([
            {"Metric":"Deals","Window":"A (Referral)","Value":resA["Deals"]},
            {"Metric":"Deals","Window":"B (Non-Referral)","Value":resB["Deals"]},
            {"Metric":"Enrollments","Window":"A (Referral)","Value":resA["Enrolments"]},
            {"Metric":"Enrollments","Window":"B (Non-Referral)","Value":resB["Enrolments"]},
            {"Metric":"Trial Scheduled","Window":"A (Referral)","Value":resA["FirstCal"]},
            {"Metric":"Trial Scheduled","Window":"B (Non-Referral)","Value":resB["FirstCal"]},
            {"Metric":"Trial Rescheduled","Window":"A (Referral)","Value":resA["CalResch"]},
            {"Metric":"Trial Rescheduled","Window":"B (Non-Referral)","Value":resB["CalResch"]},
            {"Metric":"Calibration Done","Window":"A (Referral)","Value":resA["CalDone"]},
            {"Metric":"Calibration Done","Window":"B (Non-Referral)","Value":resB["CalDone"]},
            {"Metric":"Referral generated (JetLearn Deal Source = referrals)","Window":"A (Referral)","Value":resA["ReferralGenerated"]},
            {"Metric":"Referral generated (JetLearn Deal Source = referrals)","Window":"B (Non-Referral)","Value":resB["ReferralGenerated"]},
            {"Metric":"Self-generated referrals (Referral Intent Source = Sales Generated)","Window":"A (Referral)","Value":resA["SelfGenReferrals"]},
            {"Metric":"Self-generated referrals (Referral Intent Source = Sales Generated)","Window":"B (Non-Referral)","Value":resB["SelfGenReferrals"]},
        ])
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Metric:N", title=None),
                y=alt.Y("Value:Q", title="Count"),
                column=alt.Column("Window:N", title=None),
                tooltip=["Metric","Window","Value"]
            ).properties(height=300),
            use_container_width=True
        )

        tbl = pd.DataFrame([
            {"Side":"A (Referral)",
             "Deals":resA["Deals"], "Enrollments":resA["Enrolments"],
             "Trial Scheduled":resA["FirstCal"], "Trial Rescheduled":resA["CalResch"], "Calibration Done":resA["CalDone"],
             "Referral generated (JetLearn Deal Source = referrals)":resA["ReferralGenerated"],
             "Self-generated referrals (Referral Intent Source = Sales Generated)":resA["SelfGenReferrals"],
             "Lead → Enrollment %": None if np.isnan(resA["L2E_pct"]) else round(resA["L2E_pct"],1)},
            {"Side":"B (Non-Referral)",
             "Deals":resB["Deals"], "Enrollments":resB["Enrolments"],
             "Trial Scheduled":resB["FirstCal"], "Trial Rescheduled":resB["CalResch"], "Calibration Done":resB["CalDone"],
             "Referral generated (JetLearn Deal Source = referrals)":resB["ReferralGenerated"],
             "Self-generated referrals (Referral Intent Source = Sales Generated)":resB["SelfGenReferrals"],
             "Lead → Enrollment %": None if np.isnan(resB["L2E_pct"]) else round(resB["L2E_pct"],1)},
        ])
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV — Referral vs Non-Referral (A vs B)",
            tbl.to_csv(index=False).encode("utf-8"),
            file_name="referral_vs_nonreferral_A_vs_B.csv",
            mime="text/csv"
        )
    except Exception as _e:
        st.error(f"Referral / No-Referral rendering error: {_e}")

try:
    import streamlit as st
    _master = master if 'master' in globals() else st.session_state.get('nav_master', 'Performance')
    _view = st.session_state.get('nav_sub', '')
    if _master == "Performance" and _view not in ("Referral / No-Referral",):
        with st.expander("Referral / No-Referral — quick open", expanded=False):
            if st.button("Open Referral / No-Referral", key="open_refnr"):
                st.session_state['nav_sub'] = "Referral / No-Referral"
                st.rerun()
except Exception:
    pass

# Invoke when selected
try:
    import streamlit as st
    _master = master if 'master' in globals() else st.session_state.get('nav_master', 'Performance')
    _view = st.session_state.get('nav_sub', '')
    if _master == "Performance" and _view == "Referral / No-Referral":
        _render_performance_referral_no_referral(
            df_f=df_f,
            create_col=create_col if 'create_col' in globals() else None,
            pay_col=pay_col if 'pay_col' in globals() else None,
            first_cal_sched_col=first_cal_sched_col if 'first_cal_sched_col' in globals() else None,
            cal_resched_col=cal_resched_col if 'cal_resched_col' in globals() else None,
            cal_done_col=cal_done_col if 'cal_done_col' in globals() else None,
            source_col=source_col if 'source_col' in globals() else None,
        )
except Exception as e:
    import streamlit as st
    st.warning(f"Referral / No-Referral view could not be rendered: {e}")


# ====================== Cohort Performance (Performance Pill) ======================
import pandas as _pd
import numpy as _np
import altair as _alt
import streamlit as st



# ======================
# Marketing ▶ Deal Detail
# ======================
def _render_marketing_deal_detail(df):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Deal Detail")

    if df is None or getattr(df, "empty", True):
        st.info("No data available."); return

    # Local copy & strip column names
    try:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        pass

    # ---- Column resolver
    def _col(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low: return low[c.lower()]
        for c in df.columns:
            for cand in candidates:
                if cand.lower() in c.lower(): return c
        return None

    deal_col = _col(df, ["Deal Name","Deal name","Name","Deal","Title"])
    create_col = _col(df, ["Create Date","Created Date","Deal Create Date","Date Created","Created On","Creation Date","Deal Created Date","Create_Date"])
    pay_col = _col(df, ["Payment Received Date","Payment Received date","Payment Received Date ","Enrollment Date","Enrolment Date","Enrolled On","Payment Date","Payment_Received_Date"])
    pay_col_fb = _col(df, ["Renewal: Payment Received Date","Renewal Payment Received Date"])
    source_col = _col(df, ["JetLearn Deal Source","Deal Source","Original source","Source","Original traffic source"])
    record_id_col = _col(df, ["Record ID","RecordID","ID"])

    if not deal_col or not create_col:
        st.error("Required columns missing (Deal Name / Create Date)."); return

    # ---- Date parsing
    def _to_dt(s: pd.Series):
        s = s.astype(str).str.strip().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        need = dt.isna()
        if need.any():
            compact = s.where(need)
            mask = compact.str.fullmatch(r"\d{8}", na=False)
            dt2 = pd.to_datetime(compact.where(mask), format="%d%m%Y", errors="coerce")
            dt = dt.fillna(dt2)
        need = dt.isna()
        if need.any():
            lead10 = s.where(need).str.slice(0,10)
            dt3 = pd.to_datetime(lead10, errors="coerce", dayfirst=True)
            dt = dt.fillna(dt3)
        return dt

    _C = _to_dt(df[create_col])
    _P = _to_dt(df[pay_col]) if pay_col else pd.Series(pd.NaT, index=df.index)
    _Pfb = _to_dt(df[pay_col_fb]) if pay_col_fb else pd.Series(pd.NaT, index=df.index)
    _P_any = _P.copy()
    nulls = _P_any.isna()
    if nulls.any():
        _P_any.loc[nulls] = _Pfb.loc[nulls]

    _DEAL = df[deal_col].astype(str).fillna("")
    _SRC = df[source_col].fillna("Unknown").astype(str) if source_col else pd.Series("Unknown", index=df.index)

    # ---- Controls
    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="dd_mode")
    today = date.today()
    preset = st.radio("Date range (by Create Date)", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, horizontal=True, key="dd_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        start = date(d.year, d.month, 1)
        end = date(d.year, d.month, monthrange(d.year, d.month)[1])
        return start, end
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        y = today - timedelta(days=1)
        start_d, end_d = y, y
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="dd_start")
        with c2: end_d   = st.date_input("End", value=today, key="dd_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Deal options in window + ALL
    in_create = _C.dt.date.between(start_d, end_d)
    deal_opts = sorted(_DEAL.loc[in_create].unique().tolist())
    deal_opts = ["ALL"] + deal_opts
    sel_deals = st.multiselect("Deal Name", deal_opts, default=["ALL"], key="dd_deal")
    if not sel_deals:
        st.info("Select at least one Deal Name."); return
    if "ALL" in sel_deals:
        sel_deals = [d for d in deal_opts if d != "ALL"]

    # Window masks
    m_create = _C.dt.date.between(start_d, end_d)
    m_pay_any = _P_any.dt.date.between(start_d, end_d)
    sel_mask = _DEAL.isin([str(x) for x in sel_deals])

    if mode == "MTD":
        enrol_mask = m_pay_any & m_create
    else:
        enrol_mask = m_pay_any

    # Sources for selection
    src_for_sel = "Unknown"
    src_set = set()
    if source_col:
        src_series = _SRC[m_create & sel_mask]
        if len(src_series):
            modes = src_series.mode()
            if not modes.empty:
                src_for_sel = modes.iloc[0]
            src_set = set(src_series.dropna().astype(str).unique().tolist())

    # Counts
    deal_count = int((m_create & sel_mask).sum())
    enrol_count = int((enrol_mask & sel_mask).sum())

    # Denominator for %
    if source_col:
        if src_set:
            denom_mask = m_create & (_SRC.isin(list(src_set)))
            denom_total = int(denom_mask.sum())
        elif src_for_sel != "Unknown":
            denom_mask = m_create & (_SRC == src_for_sel)
            denom_total = int(denom_mask.sum())
        else:
            denom_total = int(m_create.sum()) if m_create.any() else 0
    else:
        denom_total = int(m_create.sum()) if m_create.any() else 0

    pct_within_src = (deal_count / denom_total * 100.0) if denom_total else 0.0

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("JetLearn Deal Source (mode)", src_for_sel if src_set else "Unknown")
    with c2: st.metric("Deal Count (Created in window)", deal_count)
    with c3: st.metric("Enrollment Count", enrol_count)
    st.metric("Share of Selected Deal(s) within Source(s)", f"{pct_within_src:.2f}%")

    # Summary table
    row = {
        "Deal Name(s)": ", ".join(sel_deals[:10]) + (" …" if len(sel_deals) > 10 else ""),
        "JetLearn Deal Source (mode)": src_for_sel if src_set else "Unknown",
        "Deal Count (Create in window)": deal_count,
        "Enrollment Count": enrol_count,
        "% of Source (by Create)": round(pct_within_src, 2),
        "Mode": mode,
        "Window Start": start_d,
        "Window End": end_d,
    }
    if record_id_col:
        rec_ids = df.loc[sel_mask & (m_create | enrol_mask), record_id_col].astype(str).unique().tolist()
        row["Record IDs (in window)"] = ", ".join(rec_ids[:100])
    table = pd.DataFrame([row])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Deal Detail",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="deal_detail_summary.csv",
        mime="text/csv",
        key="dd_dl"
    )

    # Per-deal breakdown
    try:
        rows = []
        for dname in sel_deals:
            _mask = (_DEAL == str(dname))
            _src_i = "Unknown"
            if source_col:
                _s = _SRC[m_create & _mask]
                if len(_s):
                    _m = _s.mode()
                    if not _m.empty:
                        _src_i = _m.iloc[0]
            _deal_cnt_i = int((m_create & _mask).sum())
            _enrol_cnt_i = int((enrol_mask & _mask).sum())
            if source_col and _src_i != "Unknown":
                _den_i = int((m_create & (_SRC == _src_i)).sum())
            else:
                _den_i = int(m_create.sum()) if m_create.any() else 0
            _pct_i = (_deal_cnt_i / _den_i * 100.0) if _den_i else 0.0
            rows.append({
                "Deal Name": str(dname),
                "JetLearn Deal Source": _src_i,
                "Deal Count (Create in window)": _deal_cnt_i,
                "Enrollment Count": _enrol_cnt_i,
                "% of Source (by Create)": round(_pct_i, 2)
            })
        if rows:
            _detail_df = pd.DataFrame(rows)
            st.markdown("#### Per-deal breakdown")
            st.dataframe(_detail_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV — Deal Detail (per-deal)",
                data=_detail_df.to_csv(index=False).encode("utf-8"),
                file_name="deal_detail_per_deal.csv",
                mime="text/csv",
                key="dd_dl_per_deal"
            )
    except Exception as _e:
        st.caption(f"(Per-deal breakdown temporarily unavailable: {str(_e)})")

    # Contact availability
    st.markdown("#### Contact availability")
    parent_phone_col = _col(df, ["Parent Phone number","Parent Phone","Phone","Parent Phone Number","Parent Contact"])
    parent_email_col = _col(df, ["Parent Email","Parent Email ID","Email","Parent Email Address","Parent EmailID"])

    if not (parent_phone_col or parent_email_col):
        st.caption("Parent contact columns not found in the dataset.")
    else:
        base = st.radio(
            "Base population for counting",
            ["Create window (default)","Enrollment window","Either (Create or Enroll)"],
            index=0, horizontal=True, key="dd_contact_base"
        )

        if base == "Create window (default)":
            base_mask = m_create
        elif base == "Enrollment window":
            base_mask = enrol_mask
        else:
            base_mask = (m_create | enrol_mask)

        scope = base_mask & sel_mask
        total_rows = int(scope.sum())

        phone_yes = 0
        if parent_phone_col:
            _phone = df[parent_phone_col].astype(str).str.strip().fillna("")
            _phone_has = _phone.replace({"nan":"", "None":"", "NaN":""}).str.contains(r"\d", na=False)
            phone_yes = int((scope & _phone_has).sum())

        email_yes = 0
        if parent_email_col:
            _email = df[parent_email_col].astype(str).str.strip().fillna("")
            _email_has = _email.replace({"nan":"", "None":"", "NaN":""}).str.contains("@", na=False)
            email_yes = int((scope & _email_has).sum())

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Rows in scope", total_rows)
        with c2: st.metric("Parent Phone available", f"{phone_yes} ({(phone_yes/total_rows*100 if total_rows else 0):.1f}%)")
        with c3: st.metric("Parent Email available", f"{email_yes} ({(email_yes/total_rows*100 if total_rows else 0):.1f}%)")

        _contact_tbl = pd.DataFrame([{
            "Mode": mode,
            "Window Start": start_d,
            "Window End": end_d,
            "Base": base,
            "Rows in scope": total_rows,
            "Parent Phone available": phone_yes,
            "Parent Email available": email_yes,
        }])
        st.dataframe(_contact_tbl, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV — Contact availability",
            data=_contact_tbl.to_csv(index=False).encode("utf-8"),
            file_name="deal_detail_contact_availability.csv",
            mime="text/csv",
            key="dd_dl_contact"
        )


# ================================
# Marketing ▶ Sales Intern Funnel
# ================================
def _render_marketing_sales_intern_funnel(df):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Sales Intern Funnel")

    if df is None or getattr(df, "empty", True):
        st.info("No data available."); return

    try:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        pass

    def _col(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low: return low[c.lower()]
        for c in df.columns:
            for cand in candidates:
                if cand.lower() in c.lower(): return c
        return None

    intern_col = _col(df, [
        "Sales in turn","Sales Intern","Sales intern","Sales_Intern","Sales Intern Name",
        "Owner","Student/Academic Counsellor","Assigned To"
    ])
    create_col = _col(df, ["Create Date","Created Date","Deal Create Date","Date Created","Created On","Creation Date","Deal Created Date","Create_Date"])
    pay_col    = _col(df, ["Payment Received Date","Payment Received Date ","Payment Date","Enrollment Date","Enrolment Date","Enrolled On","Payment_Received_Date"])
    pay_fb     = _col(df, ["Renewal: Payment Received Date","Renewal Payment Received Date"])

    first_trial_col = _col(df, ["First Trial Scheduled Date","First Calibration Scheduled Date","[Trigger] - Calibration Booking Date","First Trial Date"])
    resched_col     = _col(df, ["Trial Rescheduled Date","Calibration Rescheduled Date","Rescheduled Date"])
    trial_done_col  = _col(df, ["Trial Done Date","Calibration Done Date","Trial Completed Date"])

    if not create_col:
        st.error("Create Date column not found."); return
    if not intern_col:
        st.warning("Sales Intern column not found. Falling back to ALL scope.")
        df["_intern_fallback"] = "ALL"
        intern_col = "_intern_fallback"

    def _to_dt(s: pd.Series):
        s = s.astype(str).str.strip().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        need = dt.isna()
        if need.any():
            compact = s.where(need)
            mask = compact.str.fullmatch(r"\d{8}", na=False)
            dt2 = pd.to_datetime(compact.where(mask), format="%d%m%Y", errors="coerce")
            dt = dt.fillna(dt2)
        need = dt.isna()
        if need.any():
            lead10 = s.where(need).str.slice(0,10)
            dt3 = pd.to_datetime(lead10, errors="coerce", dayfirst=True)
            dt = dt.fillna(dt3)
        return dt

    _C   = _to_dt(df[create_col])
    _P   = _to_dt(df[pay_col]) if pay_col else pd.Series(pd.NaT, index=df.index)
    _Pfb = _to_dt(df[pay_fb])  if pay_fb  else pd.Series(pd.NaT, index=df.index)
    _P_any = _P.copy()
    nulls = _P_any.isna()
    if nulls.any():
        _P_any.loc[nulls] = _Pfb.loc[nulls]

    _FT  = _to_dt(df[first_trial_col]) if first_trial_col else pd.Series(pd.NaT, index=df.index)
    _RS  = _to_dt(df[resched_col])     if resched_col     else pd.Series(pd.NaT, index=df.index)
    _TD  = _to_dt(df[trial_done_col])  if trial_done_col  else pd.Series(pd.NaT, index=df.index)

    _INTERN = df[intern_col].fillna("Unknown").astype(str)

    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="sif_mode")

    today = date.today()
    preset = st.radio("Date range (by Create Date for Deals)", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, horizontal=True, key="sif_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        start = date(d.year, d.month, 1)
        end = date(d.year, d.month, monthrange(d.year, d.month)[1])
        return start, end
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        y = today - timedelta(days=1)
        start_d, end_d = y, y
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="sif_start")
        with c2: end_d   = st.date_input("End", value=today, key="sif_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    in_create = _C.dt.date.between(start_d, end_d)
    intern_opts = sorted(_INTERN.loc[in_create].unique().tolist())
    intern_opts = ["ALL"] + intern_opts
    sel_interns = st.multiselect("Sales Intern", intern_opts, default=["ALL"], key="sif_interns")
    if not sel_interns:
        st.info("Select at least one Sales Intern."); return
    if "ALL" in sel_interns:
        sel_interns = [x for x in intern_opts if x != "ALL"]

    by_intern = _INTERN.isin(sel_interns)
    m_create  = _C.dt.date.between(start_d, end_d) & by_intern

    m_ft = _FT.dt.date.between(start_d, end_d) & by_intern if first_trial_col else pd.Series(False, index=df.index)
    m_rs = _RS.dt.date.between(start_d, end_d) & by_intern if resched_col     else pd.Series(False, index=df.index)
    m_td = _TD.dt.date.between(start_d, end_d) & by_intern if trial_done_col  else pd.Series(False, index=df.index)

    m_pay = _P_any.dt.date.between(start_d, end_d) & by_intern
    if mode == "MTD":
        m_enrol = m_pay & _C.dt.date.between(start_d, end_d)
    else:
        m_enrol = m_pay

    n_deals = int(m_create.sum())
    n_ft    = int(m_ft.sum())
    n_rs    = int(m_rs.sum())
    n_td    = int(m_td.sum())
    n_enrol = int(m_enrol.sum())
    conv    = (n_enrol / n_deals * 100.0) if n_deals else 0.0

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Deals (Create)", n_deals)
    with c2: st.metric("First Trial Scheduled", n_ft)
    with c3: st.metric("Trial Rescheduled", n_rs)
    with c4: st.metric("Trial Done", n_td)
    with c5: st.metric("Enrollments", n_enrol)
    st.metric("Deal → Enrollment %", f"{conv:.2f}%")

    try:
        import altair as alt
        funnel_df = pd.DataFrame({
            "Stage": ["Deals","First Trial Scheduled","Trial Rescheduled","Trial Done","Enrollments"],
            "Count": [n_deals, n_ft, n_rs, n_td, n_enrol]
        })
        st.altair_chart(
            alt.Chart(funnel_df).mark_bar().encode(x="Stage:N", y="Count:Q").properties(height=240),
            use_container_width=True
        )
    except Exception:
        pass

    summary_df = pd.DataFrame([{
        "Mode": mode,
        "Window Start": start_d,
        "Window End": end_d,
        "Sales Intern(s)": ", ".join(sel_interns[:10]) + (" …" if len(sel_interns) > 10 else ""),
        "Deals (Create)": n_deals,
        "First Trial Scheduled": n_ft,
        "Trial Rescheduled": n_rs,
        "Trial Done": n_td,
        "Enrollments": n_enrol,
        "Deal → Enrollment %": round(conv,2),
    }])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Sales Intern Funnel (summary)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="sales_intern_funnel_summary.csv",
        mime="text/csv",
        key="sif_dl_sum"
    )

    try:
        rows = []
        for intern in sel_interns:
            mask_i = (_INTERN == intern)
            deals_i = int((_C.dt.date.between(start_d, end_d) & mask_i).sum())
            ft_i    = int((_FT.dt.date.between(start_d, end_d) & mask_i).sum()) if first_trial_col else 0
            rs_i    = int((_RS.dt.date.between(start_d, end_d) & mask_i).sum()) if resched_col else 0
            td_i    = int((_TD.dt.date.between(start_d, end_d) & mask_i).sum()) if trial_done_col else 0

            pay_i = _P_any.dt.date.between(start_d, end_d) & mask_i
            if mode == "MTD":
                enrol_i = int((pay_i & _C.dt.date.between(start_d, end_d) & mask_i).sum())
            else:
                enrol_i = int(pay_i.sum())
            conv_i = (enrol_i / deals_i * 100.0) if deals_i else 0.0

            rows.append({
                "Sales Intern": intern,
                "Deals (Create)": deals_i,
                "First Trial Scheduled": ft_i,
                "Trial Rescheduled": rs_i,
                "Trial Done": td_i,
                "Enrollments": enrol_i,
                "Deal → Enrollment %": round(conv_i, 2),
            })
        if rows:
            per_df = pd.DataFrame(rows)
            st.markdown("#### Per–Sales-Intern breakdown")
            st.dataframe(per_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV — Sales Intern Funnel (per-intern)",
                data=per_df.to_csv(index=False).encode("utf-8"),
                file_name="sales_intern_funnel_per_intern.csv",
                mime="text/csv",
                key="sif_dl_detail"
            )
    except Exception as _e:
        st.caption(f"(Per-intern breakdown temporarily unavailable: {str(_e)})")


# ---- Dispatcher for Marketing ▸ Deal Detail
try:
    if st.session_state.get('nav_master') == "Marketing" and st.session_state.get('nav_sub') == "Deal Detail":
        _render_marketing_deal_detail(df_f)
except Exception as _e:
    import streamlit as _st
    _st.error(f"Deal Detail error: {_e}")

#

# ---- Dispatcher for Marketing ▸ Sales Intern Funnel
try:
    if st.session_state.get('nav_master') == "Marketing" and st.session_state.get('nav_sub') == "Sales Intern Funnel":
        _render_marketing_sales_intern_funnel(df_f)
except Exception as _e:
    import streamlit as _st
    _st.error(f"Sales Intern Funnel error: {_e}")



def _render_marketing_master_analysis(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    cal_done_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Master analysis")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])
    _first  = first_cal_sched_col if (first_cal_sched_col and first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
    _resch  = cal_resched_col     if (cal_resched_col     and cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
    _done   = cal_done_col        if (cal_done_col        and cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])

    if not _create:
        st.error("Create Date column not found."); return

    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ma_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="ma_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="ma_start")
        with c2: end_d   = st.date_input("End", value=today, key="ma_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
        return dt.dt.floor("D")

    st.markdown("### Choose columns to filter by")
    all_cols = [c for c in df_f.columns if isinstance(c, str)]
    selected_cols = st.multiselect("Columns", options=sorted(all_cols), default=[], key="ma_cols")

    d0 = df_f.copy()
    narrowed_cols = []
    if selected_cols:
        st.markdown("### Filter values")
        for c in selected_cols:
            vals = d0[c].dropna().astype(str).unique().tolist()
            opts = ["All"] + sorted(vals)
            chosen = st.multiselect(f"{c} values", options=opts, default=["All"], key=f"ma_vals_{c}")
            chosen_specific = [v for v in chosen if v != "All"]
            if chosen_specific:
                d0 = d0[d0[c].astype(str).isin(chosen_specific)]
                narrowed_cols.append(c)

    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)
    pay_dt    = _to_day_ts(d0[_pay])    if (_pay and _pay in d0.columns)    else pd.Series(pd.NaT, index=d0.index)
    first_dt  = _to_day_ts(d0[_first])  if (_first and _first in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    resch_dt  = _to_day_ts(d0[_resch])  if (_resch and _resch in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    done_dt   = _to_day_ts(d0[_done])   if (_done  and _done  in d0.columns) else pd.Series(pd.NaT, index=d0.index)

    c_in = create_dt.between(start_ts, end_ts)
    p_in = pay_dt.between(start_ts, end_ts)     if _pay   else pd.Series(False, index=d0.index)
    f_in = first_dt.between(start_ts, end_ts)   if _first else pd.Series(False, index=d0.index)
    r_in = resch_dt.between(start_ts, end_ts)   if _resch else pd.Series(False, index=d0.index)
    d_in = done_dt.between(start_ts, end_ts)    if _done  else pd.Series(False, index=d0.index)

    deals_created = int(c_in.sum())
    if mode == "MTD":
        enrolments       = int((p_in & c_in).sum()) if _pay   else 0
        trial_sched      = int((f_in & c_in).sum()) if _first else 0
        trial_resched    = int((r_in & c_in).sum()) if _resch else 0
        calibration_done = int((d_in & c_in).sum()) if _done  else 0
    else:
        enrolments       = int(p_in.sum()) if _pay   else 0
        trial_sched      = int(f_in.sum()) if _first else 0
        trial_resched    = int(r_in.sum()) if _resch else 0
        calibration_done = int(d_in.sum()) if _done  else 0

    conv = (enrolments / deals_created * 100.0) if deals_created else np.nan

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Deals Created", deals_created)
    with k2: st.metric("Enrollments", enrolments)
    with k3: st.metric("Trial Scheduled", trial_sched)
    with k4: st.metric("Trial Rescheduled", trial_resched)
    with k5: st.metric("Calibration Done", calibration_done)
    with k6: st.metric("Deal → Enrollment %", f"{conv:.1f}%" if pd.notna(conv) else "—")

    summary_table = pd.DataFrame([{
        "Deals Created": deals_created,
        "Enrollments": enrolments,
        "Trial Scheduled": trial_sched,
        "Trial Rescheduled": trial_resched,
        "Calibration Done": calibration_done,
        "Deal → Enrollment %": round(conv, 1) if pd.notna(conv) else np.nan
    }])
    st.dataframe(summary_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Master analysis (overall)",
        data=summary_table.to_csv(index=False).encode("utf-8"),
        file_name="marketing_master_analysis_overall.csv",
        mime="text/csv",
        key="dl_master_analysis_overall_csv"
    )

    auto_breakdown = len(narrowed_cols) > 0
    show_breakdown = st.toggle("Show grouped breakdown by selected values", value=auto_breakdown, key="ma_show_grp_auto")

    if show_breakdown and (selected_cols or narrowed_cols):
        grp_cols = narrowed_cols if narrowed_cols else [c for c in selected_cols if c in d0.columns]
        if grp_cols:
            g = d0.copy()
            g["_create_dt"] = create_dt
            if _pay:   g["_pay_dt"]   = pay_dt
            if _first: g["_first_dt"] = first_dt
            if _resch: g["_resch_dt"] = resch_dt
            if _done:  g["_done_dt"]  = done_dt

            def _cnt(mask: pd.Series):
                return g.loc[mask].groupby(grp_cols, dropna=False).size()

            m_create = g["_create_dt"].between(start_ts, end_ts)

            if mode == "MTD":
                m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
                m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
                m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
                m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
            else:
                m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
                m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
                m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
                m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

            gc = _cnt(m_create)
            ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
            gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
            gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
            gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

            out = pd.DataFrame(gc.rename("Deals Created"))
            if not ge.empty: out = out.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
            if not gf.empty: out = out.merge(gf.rename("Trial Scheduled"), left_index=True, right_index=True, how="left")
            if not gr.empty: out = out.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
            if not gd.empty: out = out.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
            out = out.fillna(0)

            if "Deals Created" in out.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["Deal → Enrollment %"] = np.where(
                        out["Deals Created"] != 0,
                        (out.get("Enrollments", 0) / out["Deals Created"]) * 100.0,
                        np.nan
                    ).round(1)

            out = out.reset_index()
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download CSV — Master analysis (grouped values)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="marketing_master_analysis_grouped_values.csv",
                mime="text/csv",
                key="dl_master_analysis_grouped_values_csv"
            )



def _render_marketing_referral_tracking(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    cal_done_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Referral Tracking")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_col or not parent_email_col:
        st.error("Required email columns not found. Need both 'Deal referred by (Email)' and 'Parent Email'."); return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])
    _first  = first_cal_sched_col if (first_cal_sched_col and first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
    _resch  = cal_resched_col     if (cal_resched_col     and cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
    _done   = cal_done_col        if (cal_done_col        and cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])

    if not _create:
        st.error("Create Date column not found."); return

    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rt_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rt_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rt_start")
        with c2: end_d   = st.date_input("End", value=today, key="rt_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)

    def _norm_email(x):
        if pd.isna(x): return ""
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
        return dt.dt.floor("D")

    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    ref_norm_all    = df_f[ref_col].map(_norm_email)

    existing_parent_set = set(parent_norm_all[parent_norm_all != ""].unique().tolist())
    is_referral_linked_all = ref_norm_all.apply(lambda e: (e != "") and (e in existing_parent_set))

    parent_enroll_toggle = st.toggle("Limit 'Parent Enrolled' check to selected date range (off = lifetime)", value=False, key="rt_parent_enroll_window")

    pay_dt_all = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    parent_enrolled_lifetime = set(parent_norm_all[(parent_norm_all != "") & pay_dt_all.notna()].unique().tolist()) if _pay else set()
    parent_enrolled_window   = set(parent_norm_all[(parent_norm_all != "") & pay_dt_all.between(start_ts, end_ts)].unique().tolist()) if _pay else set()
    parent_enrolled_set = parent_enrolled_window if parent_enroll_toggle else parent_enrolled_lifetime

    d0 = df_f.loc[is_referral_linked_all].copy()
    ref_norm = d0[ref_col].map(_norm_email)

    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)
    pay_dt    = _to_day_ts(d0[_pay])    if (_pay and _pay in d0.columns)    else pd.Series(pd.NaT, index=d0.index)
    first_dt  = _to_day_ts(d0[_first])  if (_first and _first in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    resch_dt  = _to_day_ts(d0[_resch])  if (_resch and _resch in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    done_dt   = _to_day_ts(d0[_done])   if (_done  and _done  in d0.columns) else pd.Series(pd.NaT, index=d0.index)

    c_in = create_dt.between(start_ts, end_ts)
    p_in = pay_dt.between(start_ts, end_ts)     if _pay   else pd.Series(False, index=d0.index)
    f_in = first_dt.between(start_ts, end_ts)   if _first else pd.Series(False, index=d0.index)
    r_in = resch_dt.between(start_ts, end_ts)   if _resch else pd.Series(False, index=d0.index)
    d_in = done_dt.between(start_ts, end_ts)    if _done  else pd.Series(False, index=d0.index)

    deals_created_all = int(c_in.sum())
    if mode == "MTD":
        enrolments_all       = int((p_in & c_in).sum()) if _pay   else 0
        trial_sched_all      = int((f_in & c_in).sum()) if _first else 0
        trial_resched_all    = int((r_in & c_in).sum()) if _resch else 0
        calibration_done_all = int((d_in & c_in).sum()) if _done  else 0
    else:
        enrolments_all       = int(p_in.sum()) if _pay   else 0
        trial_sched_all      = int(f_in.sum()) if _first else 0
        trial_resched_all    = int(r_in.sum()) if _resch else 0
        calibration_done_all = int(d_in.sum()) if _done  else 0

    conv_all = (enrolments_all / deals_created_all * 100.0) if deals_created_all else np.nan

    by_enrolled_parent = ref_norm.apply(lambda e: e in parent_enrolled_set)
    d1 = d0.loc[by_enrolled_parent].copy()

    create_dt1 = create_dt.loc[by_enrolled_parent]
    pay_dt1    = pay_dt.loc[by_enrolled_parent]    if _pay   else pd.Series(pd.NaT, index=d1.index)
    first_dt1  = first_dt.loc[by_enrolled_parent]  if _first else pd.Series(pd.NaT, index=d1.index)
    resch_dt1  = resch_dt.loc[by_enrolled_parent]  if _resch else pd.Series(pd.NaT, index=d1.index)
    done_dt1   = done_dt.loc[by_enrolled_parent]   if _done  else pd.Series(pd.NaT, index=d1.index)

    c1_in = create_dt1.between(start_ts, end_ts)
    p1_in = pay_dt1.between(start_ts, end_ts)     if _pay   else pd.Series(False, index=d1.index)
    f1_in = first_dt1.between(start_ts, end_ts)   if _first else pd.Series(False, index=d1.index)
    r1_in = resch_dt1.between(start_ts, end_ts)   if _resch else pd.Series(False, index=d1.index)
    d1_in = done_dt1.between(start_ts, end_ts)    if _done  else pd.Series(False, index=d1.index)

    deals_created_enrolled = int(c1_in.sum())
    if mode == "MTD":
        enrolments_enrolled       = int((p1_in & c1_in).sum()) if _pay   else 0
        trial_sched_enrolled      = int((f1_in & c1_in).sum()) if _first else 0
        trial_resched_enrolled    = int((r1_in & c1_in).sum()) if _resch else 0
        calibration_done_enrolled = int((d1_in & c1_in).sum()) if _done  else 0
    else:
        enrolments_enrolled       = int(p1_in.sum()) if _pay   else 0
        trial_sched_enrolled      = int(f1_in.sum()) if _first else 0
        trial_resched_enrolled    = int(r1_in.sum()) if _resch else 0
        calibration_done_enrolled = int(d1_in.sum()) if _done  else 0

    pct_referred_from_enrolled_parents = (deals_created_enrolled / deals_created_all * 100.0) if deals_created_all else np.nan
    conv_enrolled = (enrolments_enrolled / deals_created_enrolled * 100.0) if deals_created_enrolled else np.nan

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Referred Deals (All)", deals_created_all)
    with k2: st.metric("Enrollments (All)", enrolments_all)
    with k3: st.metric("Cal Scheduled (All)", trial_sched_all)
    with k4: st.metric("Cal Rescheduled (All)", trial_resched_all)
    with k5: st.metric("Cal Done (All)", calibration_done_all)
    with k6: st.metric("Deal → Enrollment % (All)", f"{conv_all:.1f}%" if pd.notna(conv_all) else "—")

    k7,k8,k9,k10,k11,k12 = st.columns(6)
    with k7: st.metric("Referred Deals (Parent Enrolled)", deals_created_enrolled)
    with k8: st.metric("% of Referred from Enrolled Parents", f"{pct_referred_from_enrolled_parents:.1f}%" if pd.notna(pct_referred_from_enrolled_parents) else "—")
    with k9: st.metric("Enrollments (Parent Enrolled)", enrolments_enrolled)
    with k10: st.metric("Cal Scheduled (Parent Enrolled)", trial_sched_enrolled)
    with k11: st.metric("Cal Rescheduled (Parent Enrolled)", trial_resched_enrolled)
    with k12: st.metric("Deal → Enrollment % (Parent Enrolled)", f"{conv_enrolled:.1f}%" if pd.notna(conv_enrolled) else "—")

    table = pd.DataFrame([{
        "Referred Deals (All)": deals_created_all,
        "Enrollments (All)": enrolments_all,
        "Cal Scheduled (All)": trial_sched_all,
        "Cal Rescheduled (All)": trial_resched_all,
        "Cal Done (All)": calibration_done_all,
        "Deal → Enrollment % (All)": round((enrolments_all / deals_created_all * 100.0), 1) if deals_created_all else np.nan,
        "Referred Deals (Parent Enrolled)": deals_created_enrolled,
        "% Referred from Enrolled Parents": round(pct_referred_from_enrolled_parents, 1) if pd.notna(pct_referred_from_enrolled_parents) else np.nan,
        "Enrollments (Parent Enrolled)": enrolments_enrolled,
        "Cal Scheduled (Parent Enrolled)": trial_sched_enrolled,
        "Cal Rescheduled (Parent Enrolled)": trial_resched_enrolled,
        "Cal Done (Parent Enrolled)": calibration_done_enrolled,
        "Deal → Enrollment % (Parent Enrolled)": round((enrolments_enrolled / deals_created_enrolled * 100.0), 1) if deals_created_enrolled else np.nan,
        "Parent Enrolled Basis": "In-range" if st.session_state.get("rt_parent_enroll_window", False) else "Lifetime"
    }])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Referral Tracking (summary)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="marketing_referral_tracking_summary.csv",
        mime="text/csv",
        key="dl_referral_tracking_summary_csv"
    )

    # ---- First enrollment maps (lifetime + in-window) ----
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt_all"]  = pay_dt_all
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt_all"].notna()].groupby("_parent_norm")["_pay_dt_all"].min()
    first_pay_win = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt_all"].between(start_ts, end_ts)].groupby("_parent_norm")["_pay_dt_all"].min()

    # ---- Breakdown: by Referrer Email ----
    show_by_email = st.toggle("Show breakdown by Referrer Email", value=True, key="rt_show_email")
    if show_by_email:
        g = d0.copy()
        g["_ref_email_norm"] = ref_norm
        g["_create_dt"] = create_dt
        if _pay:   g["_pay_dt"]   = pay_dt
        if _first: g["_first_dt"] = first_dt
        if _resch: g["_resch_dt"] = resch_dt
        if _done:  g["_done_dt"]  = done_dt

        grp = ["_ref_email_norm"]

        def _cnt(mask: pd.Series):
            return g.loc[mask].groupby(grp, dropna=False).size()

        m_create = g["_create_dt"].between(start_ts, end_ts)
        if mode == "MTD":
            m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
            m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
            m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
            m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
        else:
            m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
            m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
            m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
            m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

        gc = _cnt(m_create)
        ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
        gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
        gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
        gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

        out = pd.DataFrame(gc.rename("Deals Created"))
        if not ge.empty: out = out.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
        if not gf.empty: out = out.merge(gf.rename("Calibration Scheduled"), left_index=True, right_index=True, how="left")
        if not gr.empty: out = out.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
        if not gd.empty: out = out.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
        out = out.fillna(0)

        if "Deals Created" in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["Deal → Enrollment %"] = np.where(
                    out["Deals Created"] != 0,
                    (out.get("Enrollments", 0) / out["Deals Created"]) * 100.0,
                    np.nan
                ).round(1)

        chosen_first_map = first_pay_win if st.session_state.get("rt_parent_enroll_window", False) else first_pay_any
        enrolled_set = parent_enrolled_window if st.session_state.get("rt_parent_enroll_window", False) else parent_enrolled_lifetime
        out = out.reset_index().rename(columns={"_ref_email_norm":"Referrer Email"})
        out["Parent Enrolled?"] = out["Referrer Email"].map(lambda e: "Yes" if (str(e).lower() in enrolled_set) else "No")
        out["First Parent Enrollment Date"] = out["Referrer Email"].map(lambda e: chosen_first_map.get(str(e).lower(), pd.NaT))

        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download CSV — Referral Tracking by Referrer Email",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_tracking_by_referrer.csv",
            mime="text/csv",
            key="dl_referral_tracking_by_referrer_csv"
        )

    # ---- New View: Referrer purchase timing (±N days) ----
    st.markdown("### Referrer purchase timing (±N days)")
    offset_days = st.number_input("Offset days", min_value=1, max_value=365, value=45, step=1, key="rt_timing_offset")
    offset = pd.Timedelta(days=int(offset_days))
    win_lo = start_ts - offset
    win_hi = end_ts + offset

    # Use lifetime first enrollment date relative to window
    first_enroll_map = first_pay_any

    def _bucket(d):
        if pd.isna(d): return "No enrollment"
        if win_lo <= d < start_ts: return f"Before-{int(offset_days)}"
        if start_ts <= d <= end_ts: return "In-window"
        if end_ts < d <= win_hi: return f"After-{int(offset_days)}"
        if d < win_lo: return "Earlier than offset"
        return "Later than offset"

    ref_first_enroll = ref_norm.map(lambda e: first_enroll_map.get(e, pd.NaT))
    ref_bucket = ref_first_enroll.map(_bucket)

    primary_buckets = [f"Before-{int(offset_days)}", "In-window", f"After-{int(offset_days)}"]
    cols = st.columns(len(primary_buckets))
    for i, b in enumerate(primary_buckets):
        mask_b = (ref_bucket == b)
        deals_b = int((c_in & mask_b).sum())
        with cols[i]:
            st.metric(b, deals_b)

    g = d0.copy()
    g["_ref_email_norm"] = ref_norm
    g["_bucket"] = ref_bucket
    g["_create_dt"] = create_dt
    if _pay:   g["_pay_dt"]   = pay_dt
    if _first: g["_first_dt"] = first_dt
    if _resch: g["_resch_dt"] = resch_dt
    if _done:  g["_done_dt"]  = done_dt

    grp = ["_ref_email_norm","_bucket"]

    def _cnt(mask: pd.Series):
        return g.loc[mask].groupby(grp, dropna=False).size()

    m_create = g["_create_dt"].between(start_ts, end_ts)
    if mode == "MTD":
        m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
        m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
        m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
        m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
    else:
        m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
        m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
        m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
        m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

    gc = _cnt(m_create)
    ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
    gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
    gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
    gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

    out_t = pd.DataFrame(gc.rename("Deals Created"))
    if not ge.empty: out_t = out_t.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
    if not gf.empty: out_t = out_t.merge(gf.rename("Calibration Scheduled"), left_index=True, right_index=True, how="left")
    if not gr.empty: out_t = out_t.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
    if not gd.empty: out_t = out_t.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
    out_t = out_t.fillna(0)

    if "Deals Created" in out_t.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out_t["Deal → Enrollment %"] = np.where(
                out_t["Deals Created"] != 0,
                (out_t.get("Enrollments", 0) / out_t["Deals Created"]) * 100.0,
                np.nan
            ).round(1)

    out_t = out_t.reset_index().rename(columns={"_ref_email_norm":"Referrer Email","_bucket":"Timing Bucket"})
    out_t["First Parent Enrollment Date"] = out_t["Referrer Email"].map(lambda e: first_pay_any.get(str(e).lower(), pd.NaT))

    st.dataframe(out_t, use_container_width=True)
    st.download_button(
        "Download CSV — Referrer timing breakdown",
        data=out_t.to_csv(index=False).encode("utf-8"),
        file_name="marketing_referral_referrer_timing_breakdown.csv",
        mime="text/csv",
        key="dl_referral_timing_breakdown_csv"
    )

    # ---- Details expander ----
    with st.expander("Show referred deal details"):
        only_enrolled_toggle = st.checkbox("Show only deals referred by enrolled parents", value=False, key="rt_only_enrolled")
        base_df = d1 if only_enrolled_toggle else d0
        cols = []
        for c in [ref_col, parent_email_col, _create, _first, _resch, _done, _pay, "Deal Name","Record ID","Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source"]:
            if c and c in base_df.columns and c not in cols:
                cols.append(c)
        detail = base_df.loc[:, cols].copy()
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Referred deal details",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_tracking_details.csv",
            mime="text/csv",
            key="dl_referral_tracking_details_csv"
        )



# --- Router: Marketing → Master analysis ---
try:
    _master_for_ma = master if 'master' in globals() else st.session_state.get('nav_master', '')
    _view_for_ma = view if 'view' in globals() else st.session_state.get('nav_sub', '')
    if _master_for_ma == "Marketing" and _view_for_ma == "Master analysis":
        _render_marketing_master_analysis(
            df_f=df_f,
            create_col=create_col,
            pay_col=pay_col,
            first_cal_sched_col=first_cal_sched_col,
            cal_resched_col=cal_resched_col,
            cal_done_col=cal_done_col,
        )
except Exception as _e_ma:
    import streamlit as st
    st.error(f"Master analysis failed: {type(_e_ma).__name__}: {_e_ma}")

# --- Router: Marketing → Referral Tracking ---
try:
    _master_for_rt = master if 'master' in globals() else st.session_state.get('nav_master', '')
    _view_for_rt = view if 'view' in globals() else st.session_state.get('nav_sub', '')
    if _master_for_rt == "Marketing" and _view_for_rt == "Referral Tracking":
        _render_marketing_referral_tracking(
            df_f=df_f,
            create_col=create_col,
            pay_col=pay_col,
            first_cal_sched_col=first_cal_sched_col,
            cal_resched_col=cal_resched_col,
            cal_done_col=cal_done_col,
        )
except Exception as _e_rt:
    import streamlit as st
    st.error(f"Referral Tracking failed: {type(_e_rt).__name__}: {_e_rt}")



def _render_marketing_referral_tracking(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    cal_done_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Referral Tracking")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_col or not parent_email_col:
        st.error("Required email columns not found. Need both 'Deal referred by (Email)' and 'Parent Email'."); return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])
    _first  = first_cal_sched_col if (first_cal_sched_col and first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
    _resch  = cal_resched_col     if (cal_resched_col     and cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
    _done   = cal_done_col        if (cal_done_col        and cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])

    if not _create:
        st.error("Create Date column not found."); return

    # ---- Mode & Date presets ----
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rt_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rt_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rt_start")
        with c2: end_d   = st.date_input("End", value=today, key="rt_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)

    def _norm_email(x):
        if pd.isna(x): return ""
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
        return dt.dt.floor("D")

    # ---- Normalize emails ----
    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    ref_norm_all    = df_f[ref_col].map(_norm_email)

    # Existing parents set & referral-linked deals
    existing_parent_set = set(parent_norm_all[parent_norm_all != ""].unique().tolist())
    is_referral_linked_all = ref_norm_all.apply(lambda e: (e != "") and (e in existing_parent_set))

    # Parent enrollment basis (not directly used in the new KPIs but retained for other blocks)
    parent_enroll_toggle = st.toggle("Limit 'Parent Enrolled' check to selected date range (off = lifetime)", value=False, key="rt_parent_enroll_window")

    # Parse all payment dates (for building first-enroll maps)
    pay_dt_all = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    parent_enrolled_lifetime = set(parent_norm_all[(parent_norm_all != "") & pay_dt_all.notna()].unique().tolist()) if _pay else set()
    parent_enrolled_window   = set(parent_norm_all[(parent_norm_all != "") & pay_dt_all.between(start_ts, end_ts)].unique().tolist()) if _pay else set()

    # ---- Filter to referral-linked new deals ----
    d0 = df_f.loc[is_referral_linked_all].copy()
    ref_norm = d0[ref_col].map(_norm_email)

    # ---- Parse event dates for referred deals ----
    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)
    pay_dt    = _to_day_ts(d0[_pay])    if (_pay and _pay in d0.columns)    else pd.Series(pd.NaT, index=d0.index)
    first_dt  = _to_day_ts(d0[_first])  if (_first and _first in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    resch_dt  = _to_day_ts(d0[_resch])  if (_resch and _resch in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    done_dt   = _to_day_ts(d0[_done])   if (_done  and _done  in d0.columns) else pd.Series(pd.NaT, index=d0.index)

    # ---- Window masks for deal-led metrics ----
    c_in = create_dt.between(start_ts, end_ts)
    p_in = pay_dt.between(start_ts, end_ts)     if _pay   else pd.Series(False, index=d0.index)
    f_in = first_dt.between(start_ts, end_ts)   if _first else pd.Series(False, index=d0.index)
    r_in = resch_dt.between(start_ts, end_ts)   if _resch else pd.Series(False, index=d0.index)
    d_in = done_dt.between(start_ts, end_ts)    if _done  else pd.Series(False, index=d0.index)

    # ---- Base totals (all referred) ----
    deals_created_all = int(c_in.sum())
    if mode == "MTD":
        enrolments_all       = int((p_in & c_in).sum()) if _pay   else 0
        trial_sched_all      = int((f_in & c_in).sum()) if _first else 0
        trial_resched_all    = int((r_in & c_in).sum()) if _resch else 0
        calibration_done_all = int((d_in & c_in).sum()) if _done  else 0
    else:
        enrolments_all       = int(p_in.sum()) if _pay   else 0
        trial_sched_all      = int(f_in.sum()) if _first else 0
        trial_resched_all    = int(r_in.sum()) if _resch else 0
        calibration_done_all = int(d_in.sum()) if _done  else 0

    conv_all = (enrolments_all / deals_created_all * 100.0) if deals_created_all else np.nan

    # ---- First enrollment map (lifetime) for parents/referrers ----
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt_all"]  = pay_dt_all
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt_all"].notna()].groupby("_parent_norm")["_pay_dt_all"].min()

    # ====================== NEW KPIs: Referrer already-enrolled window ======================
    st.markdown("### Referrer already-enrolled window at deal creation")
    offset_days = st.number_input("Offset days for 'already-enrolled' window", min_value=1, max_value=365, value=45, step=1, key="rt_already_window")

    # For each referred deal, get the referrer's first enrollment date
    ref_first_enroll_dt = ref_norm.map(lambda e: first_pay_any.get(e, pd.NaT))

    # Only consider deals created in-window (we always count deals from c_in)
    base_mask = c_in.copy()

    # A referrer is "already enrolled" for that deal if their first enroll date <= the deal's create date
    already_mask = ref_first_enroll_dt.notna() & (ref_first_enroll_dt <= create_dt)

    # Compute the gap in days (deal_create - first_enroll)
    gap_days = (create_dt - ref_first_enroll_dt).dt.days

    within_mask = already_mask & (gap_days <= int(offset_days)) & (gap_days >= 0)
    before_mask = already_mask & (gap_days > int(offset_days))

    # Optional: after and never (not shown as KPIs unless toggled later)
    after_mask = ref_first_enroll_dt.notna() & (ref_first_enroll_dt > create_dt)
    never_mask = ref_first_enroll_dt.isna()

    # Counts within the selected window
    deals_already_within = int((base_mask & within_mask).sum())
    deals_already_before = int((base_mask & before_mask).sum())

    kx1, kx2 = st.columns(2)
    with kx1: st.metric(f"Ref. Deals — referrer enrolled ≤{int(offset_days)}d before", deals_already_within)
    with kx2: st.metric(f"Ref. Deals — referrer enrolled >{int(offset_days)}d before", deals_already_before)

    # Breakdown by Referrer Email for these two buckets
    show_already_breakdown = st.toggle("Show breakdown for 'already-enrolled' buckets", value=True, key="rt_show_already_breakdown")
    if show_already_breakdown:
        g = d0.copy()
        g["_ref_email_norm"] = ref_norm
        g["_create_dt"] = create_dt
        g["_within"] = within_mask
        g["_before"] = before_mask
        g["_base"] = base_mask
        if _pay:   g["_pay_dt"]   = pay_dt
        if _first: g["_first_dt"] = first_dt
        if _resch: g["_resch_dt"] = resch_dt
        if _done:  g["_done_dt"]  = done_dt

        # Build two grouped outputs and stack
        def _build_bucket(label, flag_series):
            grp = ["_ref_email_norm"]
            def _cnt(mask):
                return g.loc[mask].groupby(grp, dropna=False).size()
            m_create = g["_create_dt"].between(start_ts, end_ts) & flag_series & g["_base"]
            if mode == "MTD":
                m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
                m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
                m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
                m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
            else:
                m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
                m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
                m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
                m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

            gc = _cnt(m_create)
            ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
            gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
            gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
            gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

            out = pd.DataFrame(gc.rename("Deals Created"))
            if not ge.empty: out = out.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
            if not gf.empty: out = out.merge(gf.rename("Calibration Scheduled"), left_index=True, right_index=True, how="left")
            if not gr.empty: out = out.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
            if not gd.empty: out = out.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
            out = out.fillna(0)

            if "Deals Created" in out.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["Deal → Enrollment %"] = np.where(
                        out["Deals Created"] != 0,
                        (out.get("Enrollments", 0) / out["Deals Created"]) * 100.0,
                        np.nan
                    ).round(1)
            out = out.reset_index().rename(columns={"_ref_email_norm":"Referrer Email"})
            out.insert(1, "Already-enrolled bucket", label)
            return out

        out_within = _build_bucket(f"≤{int(offset_days)}d before", g["_within"])
        out_before = _build_bucket(f">{int(offset_days)}d before", g["_before"])
        out_combo = pd.concat([out_within, out_before], axis=0, ignore_index=True)

        # Attach first enrollment date for readability
        out_combo["First Parent Enrollment Date"] = out_combo["Referrer Email"].map(lambda e: first_pay_any.get(str(e).lower(), pd.NaT))

        st.dataframe(out_combo, use_container_width=True)
        st.download_button(
            "Download CSV — Already-enrolled buckets by Referrer Email",
            data=out_combo.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_already_enrolled_buckets_by_referrer.csv",
            mime="text/csv",
            key="dl_referral_already_buckets_csv"
        )

    # ====== (Keep existing Timing view based on window vs parent enrollment date) ======
    st.markdown("---")
    st.markdown("### Referrer purchase timing (±N days)")
    timing_days = st.number_input("Offset days", min_value=1, max_value=365, value=45, step=1, key="rt_timing_offset")
    offset = pd.Timedelta(days=int(timing_days))
    win_lo = start_ts - offset
    win_hi = end_ts + offset

    def _bucket(d):
        if pd.isna(d): return "No enrollment"
        if win_lo <= d < start_ts: return f"Before-{int(timing_days)}"
        if start_ts <= d <= end_ts: return "In-window"
        if end_ts < d <= win_hi: return f"After-{int(timing_days)}"
        if d < win_lo: return "Earlier than offset"
        return "Later than offset"

    ref_first_enroll = ref_norm.map(lambda e: first_pay_any.get(e, pd.NaT))
    ref_bucket = ref_first_enroll.map(_bucket)

    primary_buckets = [f"Before-{int(timing_days)}", "In-window", f"After-{int(timing_days)}"]
    cols = st.columns(len(primary_buckets))
    for i, b in enumerate(primary_buckets):
        mask_b = (ref_bucket == b)
        deals_b = int((c_in & mask_b).sum())
        with cols[i]:
            st.metric(b, deals_b)

    g = d0.copy()
    g["_ref_email_norm"] = ref_norm
    g["_bucket"] = ref_bucket
    g["_create_dt"] = create_dt
    if _pay:   g["_pay_dt"]   = pay_dt
    if _first: g["_first_dt"] = first_dt
    if _resch: g["_resch_dt"] = resch_dt
    if _done:  g["_done_dt"]  = done_dt

    grp = ["_ref_email_norm","_bucket"]
    def _cnt(mask: pd.Series):
        return g.loc[mask].groupby(grp, dropna=False).size()

    m_create = g["_create_dt"].between(start_ts, end_ts)
    if mode == "MTD":
        m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
        m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
        m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
        m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
    else:
        m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
        m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
        m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
        m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

    gc = _cnt(m_create)
    ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
    gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
    gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
    gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

    out_t = pd.DataFrame(gc.rename("Deals Created"))
    if not ge.empty: out_t = out_t.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
    if not gf.empty: out_t = out_t.merge(gf.rename("Calibration Scheduled"), left_index=True, right_index=True, how="left")
    if not gr.empty: out_t = out_t.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
    if not gd.empty: out_t = out_t.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
    out_t = out_t.fillna(0)

    if "Deals Created" in out_t.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out_t["Deal → Enrollment %"] = np.where(
                out_t["Deals Created"] != 0,
                (out_t.get("Enrollments", 0) / out_t["Deals Created"]) * 100.0,
                np.nan
            ).round(1)

    out_t = out_t.reset_index().rename(columns={"_ref_email_norm":"Referrer Email","_bucket":"Timing Bucket"})
    out_t["First Parent Enrollment Date"] = out_t["Referrer Email"].map(lambda e: first_pay_any.get(str(e).lower(), pd.NaT))

    st.dataframe(out_t, use_container_width=True)
    st.download_button(
        "Download CSV — Referrer timing breakdown",
        data=out_t.to_csv(index=False).encode("utf-8"),
        file_name="marketing_referral_referrer_timing_breakdown.csv",
        mime="text/csv",
        key="dl_referral_timing_breakdown_csv"
    )

    # ---- Details expander ----
    with st.expander("Show referred deal details"):
        only_enrolled_toggle = st.checkbox("Show only deals referred by enrolled parents", value=False, key="rt_only_enrolled")
        base_df = d0 if not only_enrolled_toggle else d0[ref_norm.isin(parent_enrolled_lifetime)]
        cols = []
        for c in [ref_col, parent_email_col, _create, _first, _resch, _done, _pay, "Deal Name","Record ID","Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source"]:
            if c and c in base_df.columns and c not in cols:
                cols.append(c)
        detail = base_df.loc[:, cols].copy()
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Referred deal details",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_tracking_details.csv",
            mime="text/csv",
            key="dl_referral_tracking_details_csv"
        )



def _render_marketing_referral_tracking(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    cal_done_col: str | None,
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Referral Tracking")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    # ---- Resolve columns ----
    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_col or not parent_email_col:
        st.error("Required email columns not found. Need both 'Deal referred by (Email)' and 'Parent Email'."); return

    # ---- Date columns ----
    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])
    _first  = first_cal_sched_col if (first_cal_sched_col and first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
    _resch  = cal_resched_col     if (cal_resched_col     and cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
    _done   = cal_done_col        if (cal_done_col        and cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])

    if not _create:
        st.error("Create Date column not found."); return

    # ---- Mode & Date presets for deal selection ----
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rt_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rt_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rt_start")
        with c2: end_d   = st.date_input("End", value=today, key="rt_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Normalize bounds to naive Timestamps at midnight
    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)
    today_ts = pd.Timestamp(today).normalize().tz_localize(None)

    # ---- Helpers ----
    def _norm_email(x):
        if pd.isna(x): return ""
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
        return dt.dt.floor("D")

    # ---- Normalize emails ----
    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    ref_norm_all    = df_f[ref_col].map(_norm_email)

    # Existing parents set & referral-linked deals
    existing_parent_set = set(parent_norm_all[parent_norm_all != ""].unique().tolist())
    is_referral_linked_all = ref_norm_all.apply(lambda e: (e != "") and (e in existing_parent_set))

    # Parse all payment dates (for building first-enroll maps)
    pay_dt_all = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)

    # ---- Filter to referral-linked new deals ----
    d0 = df_f.loc[is_referral_linked_all].copy()
    ref_norm = d0[ref_col].map(_norm_email)

    # ---- Parse event dates for referred deals ----
    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)
    pay_dt    = _to_day_ts(d0[_pay])    if (_pay and _pay in d0.columns)    else pd.Series(pd.NaT, index=d0.index)
    first_dt  = _to_day_ts(d0[_first])  if (_first and _first in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    resch_dt  = _to_day_ts(d0[_resch])  if (_resch and _resch in d0.columns) else pd.Series(pd.NaT, index=d0.index)
    done_dt   = _to_day_ts(d0[_done])   if (_done  and _done  in d0.columns) else pd.Series(pd.NaT, index=d0.index)

    # ---- Window masks for deal-led metrics ----
    c_in = create_dt.between(start_ts, end_ts)
    p_in = pay_dt.between(start_ts, end_ts)     if _pay   else pd.Series(False, index=d0.index)
    f_in = first_dt.between(start_ts, end_ts)   if _first else pd.Series(False, index=d0.index)
    r_in = resch_dt.between(start_ts, end_ts)   if _resch else pd.Series(False, index=d0.index)
    d_in = done_dt.between(start_ts, end_ts)    if _done  else pd.Series(False, index=d0.index)

    # ---- Base totals (all referred) ----
    deals_created_all = int(c_in.sum())
    if mode == "MTD":
        enrolments_all       = int((p_in & c_in).sum()) if _pay   else 0
        trial_sched_all      = int((f_in & c_in).sum()) if _first else 0
        trial_resched_all    = int((r_in & c_in).sum()) if _resch else 0
        calibration_done_all = int((d_in & c_in).sum()) if _done  else 0
    else:
        enrolments_all       = int(p_in.sum()) if _pay   else 0
        trial_sched_all      = int(f_in.sum()) if _first else 0
        trial_resched_all    = int(r_in.sum()) if _resch else 0
        calibration_done_all = int(d_in.sum()) if _done  else 0

    conv_all = (enrolments_all / deals_created_all * 100.0) if deals_created_all else np.nan

    # ---- First enrollment map (lifetime) for parents/referrers ----
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt_all"]  = pay_dt_all
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt_all"].notna()].groupby("_parent_norm")["_pay_dt_all"].min()

    # ====================== UPDATED: Already-enrolled window (ROLLING vs TODAY) ======================
    st.markdown("### Referrer already-enrolled window (rolling vs today)")
    rolling_days = st.number_input("Rolling window (days, ending today)", min_value=1, max_value=365, value=45, step=1, key="rt_already_window_rolling")
    lo = today_ts - pd.Timedelta(days=int(rolling_days))
    hi = today_ts  # inclusive

    # Referrer lifetime first enrollment date
    ref_first_enroll_dt = ref_norm.map(lambda e: first_pay_any.get(e, pd.NaT))

    # Define rolling buckets (relative to TODAY, not the deal range)
    within_roll = ref_first_enroll_dt.between(lo, hi, inclusive="both")
    before_roll = ref_first_enroll_dt.notna() & (ref_first_enroll_dt < lo)
    # Optional informational buckets:
    after_today = ref_first_enroll_dt > hi
    never_enrolled = ref_first_enroll_dt.isna()

    # Count referred deals created within the selected deal window, split by rolling buckets
    deals_within_roll = int((c_in & within_roll).sum())
    deals_before_roll = int((c_in & before_roll).sum())

    cx1, cx2 = st.columns(2)
    with cx1: st.metric(f"Ref. Deals — referrer enrolled ≤{int(rolling_days)}d (rolling to today)", deals_within_roll)
    with cx2: st.metric(f"Ref. Deals — referrer enrolled >{int(rolling_days)}d (rolling to today)", deals_before_roll)

    # ---- Existing metrics (kept as-is) ----
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Referred Deals (All)", deals_created_all)
    with k2: st.metric("Enrollments (All)", enrolments_all)
    with k3: st.metric("Cal Scheduled (All)", trial_sched_all)
    with k4: st.metric("Cal Rescheduled (All)", trial_resched_all)
    with k5: st.metric("Cal Done (All)", calibration_done_all)
    with k6: st.metric("Deal → Enrollment % (All)", f"{conv_all:.1f}%" if pd.notna(conv_all) else "—")

    # ---- Breakdown by Referrer Email for rolling buckets ----
    show_roll_breakdown = st.toggle("Show breakdown for rolling buckets (by Referrer Email)", value=True, key="rt_show_roll_breakdown")
    if show_roll_breakdown:
        g = d0.copy()
        g["_ref_email_norm"] = ref_norm
        g["_create_dt"] = create_dt
        g["_within_roll"] = within_roll
        g["_before_roll"] = before_roll
        g["_c_in"] = c_in
        if _pay:   g["_pay_dt"]   = pay_dt
        if _first: g["_first_dt"] = first_dt
        if _resch: g["_resch_dt"] = resch_dt
        if _done:  g["_done_dt"]  = done_dt

        def _build(label, flag_series):
            grp = ["_ref_email_norm"]
            def _cnt(mask): return g.loc[mask].groupby(grp, dropna=False).size()
            m_create = g["_c_in"] & flag_series
            if mode == "MTD":
                m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
                m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
                m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
                m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
            else:
                m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
                m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
                m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
                m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

            gc = _cnt(m_create)
            ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
            gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
            gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
            gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

            out = pd.DataFrame(gc.rename("Deals Created"))
            if not ge.empty: out = out.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
            if not gf.empty: out = out.merge(gf.rename("Calibration Scheduled"), left_index=True, right_index=True, how="left")
            if not gr.empty: out = out.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
            if not gd.empty: out = out.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
            out = out.fillna(0)
            if "Deals Created" in out.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["Deal → Enrollment %"] = np.where(
                        out["Deals Created"] != 0,
                        (out.get("Enrollments", 0) / out["Deals Created"]) * 100.0,
                        np.nan
                    ).round(1)
            out = out.reset_index().rename(columns={"_ref_email_norm":"Referrer Email"})
            out.insert(1, "Rolling bucket", label)
            return out

        out_within = _build(f"≤{int(rolling_days)}d (to today)", g["_within_roll"])
        out_before = _build(f">{int(rolling_days)}d (to today)", g["_before_roll"])
        out_combo = pd.concat([out_within, out_before], axis=0, ignore_index=True)

        # Attach first enrollment date for readability
        out_combo["First Parent Enrollment Date"] = out_combo["Referrer Email"].map(lambda e: first_pay_any.get(str(e).lower(), pd.NaT))

        st.dataframe(out_combo, use_container_width=True)
        st.download_button(
            "Download CSV — Rolling buckets by Referrer Email",
            data=out_combo.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_rolling_buckets_by_referrer.csv",
            mime="text/csv",
            key="dl_referral_rolling_buckets_csv"
        )

    # ---- Keep prior "Referrer purchase timing (±N days vs window)" block for completeness ----
    st.markdown("---")
    st.markdown("### Referrer purchase timing (±N days)")
    timing_days = st.number_input("Offset days", min_value=1, max_value=365, value=45, step=1, key="rt_timing_offset")
    offset = pd.Timedelta(days=int(timing_days))
    win_lo = start_ts - offset
    win_hi = end_ts + offset

    def _bucket(d):
        if pd.isna(d): return "No enrollment"
        if win_lo <= d < start_ts: return f"Before-{int(timing_days)}"
        if start_ts <= d <= end_ts: return "In-window"
        if end_ts < d <= win_hi: return f"After-{int(timing_days)}"
        if d < win_lo: return "Earlier than offset"
        return "Later than offset"

    ref_first_enroll = ref_norm.map(lambda e: first_pay_any.get(e, pd.NaT))
    ref_bucket = ref_first_enroll.map(_bucket)

    primary_buckets = [f"Before-{int(timing_days)}", "In-window", f"After-{int(timing_days)}"]
    cols = st.columns(len(primary_buckets))
    for i, b in enumerate(primary_buckets):
        mask_b = (ref_bucket == b)
        deals_b = int((c_in & mask_b).sum())
        with cols[i]:
            st.metric(b, deals_b)

    g = d0.copy()
    g["_ref_email_norm"] = ref_norm
    g["_bucket"] = ref_bucket
    g["_create_dt"] = create_dt
    if _pay:   g["_pay_dt"]   = pay_dt
    if _first: g["_first_dt"] = first_dt
    if _resch: g["_resch_dt"] = resch_dt
    if _done:  g["_done_dt"]  = done_dt

    grp = ["_ref_email_norm","_bucket"]
    def _cnt(mask: pd.Series):
        return g.loc[mask].groupby(grp, dropna=False).size()

    m_create = g["_create_dt"].between(start_ts, end_ts)
    if mode == "MTD":
        m_enrol = (g.get("_pay_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _pay else None
        m_first = (g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _first else None
        m_resch = (g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _resch else None
        m_done  = (g.get("_done_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) & m_create) if _done else None
    else:
        m_enrol = g.get("_pay_dt",   pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _pay else None
        m_first = g.get("_first_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _first else None
        m_resch = g.get("_resch_dt", pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _resch else None
        m_done  = g.get("_done_dt",  pd.Series(index=g.index, dtype="datetime64[ns]")).between(start_ts, end_ts) if _done else None

    gc = _cnt(m_create)
    ge = _cnt(m_enrol) if m_enrol is not None else pd.Series(dtype=int)
    gf = _cnt(m_first) if m_first is not None else pd.Series(dtype=int)
    gr = _cnt(m_resch) if m_resch is not None else pd.Series(dtype=int)
    gd = _cnt(m_done)  if m_done  is not None else pd.Series(dtype=int)

    out_t = pd.DataFrame(gc.rename("Deals Created"))
    if not ge.empty: out_t = out_t.merge(ge.rename("Enrollments"), left_index=True, right_index=True, how="left")
    if not gf.empty: out_t = out_t.merge(gf.rename("Calibration Scheduled"), left_index=True, right_index=True, how="left")
    if not gr.empty: out_t = out_t.merge(gr.rename("Calibration Rescheduled"), left_index=True, right_index=True, how="left")
    if not gd.empty: out_t = out_t.merge(gd.rename("Calibration Done"), left_index=True, right_index=True, how="left")
    out_t = out_t.fillna(0)

    if "Deals Created" in out_t.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out_t["Deal → Enrollment %"] = np.where(
                out_t["Deals Created"] != 0,
                (out_t.get("Enrollments", 0) / out_t["Deals Created"]) * 100.0,
                np.nan
            ).round(1)

    out_t = out_t.reset_index().rename(columns={"_ref_email_norm":"Referrer Email","_bucket":"Timing Bucket"})
    out_t["First Parent Enrollment Date"] = out_t["Referrer Email"].map(lambda e: first_pay_any.get(str(e).lower(), pd.NaT))

    st.dataframe(out_t, use_container_width=True)
    st.download_button(
        "Download CSV — Referrer timing breakdown",
        data=out_t.to_csv(index=False).encode("utf-8"),
        file_name="marketing_referral_referrer_timing_breakdown.csv",
        mime="text/csv",
        key="dl_referral_timing_breakdown_csv"
    )

    # ---- Details expander (unchanged) ----
    with st.expander("Show referred deal details"):
        only_enrolled_toggle = st.checkbox("Show only deals referred by enrolled parents", value=False, key="rt_only_enrolled")
        base_df = d0 if not only_enrolled_toggle else d0[ref_norm.isin(first_pay_any.index)]
        cols = []
        for c in [ref_col, parent_email_col, _create, _first, _resch, _done, _pay, "Deal Name","Record ID","Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source"]:
            if c and c in base_df.columns and c not in cols:
                cols.append(c)
        detail = base_df.loc[:, cols].copy()
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Referred deal details",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_tracking_details.csv",
            mime="text/csv",
            key="dl_referral_tracking_details_csv"
        )



def _render_marketing_ref_tracker(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
):
    """
    Marketing → Ref_Tracker
    Goal:
      - Count and list rows where `Deal referred by(Email)` == `Parent Email` (normalized).
      - Show a rolling toggle: Parent Email enrolled within <=N days to today vs >N days before today.
      - Support MTD/Cohort counting and date-range presets: Yesterday, Today, This Month, Last Month, Custom.
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Ref_Tracker")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    # ---- Resolve necessary columns ----
    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_col or not parent_email_col:
        st.error("Required email columns not found. Need both 'Deal referred by (Email)' and 'Parent Email'.")
        return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])

    if not _create:
        st.error("Create Date column not found."); return

    # ---- Mode & Date presets ----
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rtk_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rtk_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rtk_start")
        with c2: end_d   = st.date_input("End", value=today, key="rtk_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Normalize bounds
    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)
    today_ts = pd.Timestamp(today).normalize().tz_localize(None)

    # ---- Helpers ----
    def _norm_email(x):
        if pd.isna(x): return ""
        try: return str(x).strip().lower()
        except Exception: return ""

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try: dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception: dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try: dt = dt.dt.tz_localize(None)
            except Exception: pass
        return dt.dt.floor("D")

    # ---- Normalize emails & build match mask (ref == parent) ----
    ref_norm_all    = df_f[ref_col].map(_norm_email)
    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    same_email_mask_all = (ref_norm_all != "") & (ref_norm_all == parent_norm_all)

    # Restrict to rows where emails match
    d0 = df_f.loc[same_email_mask_all].copy()

    # ---- Parse dates for filtered rows ----
    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)
    pay_dt_all = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)

    # ---- Deal window masks ----
    c_in = create_dt.between(start_ts, end_ts)

    # ---- Rolling 45d (N) buckets based on Parent Email first enrollment date (lifetime), relative to TODAY ----
    st.markdown("### Rolling parent-enrollment window (relative to today)")
    rolling_days = st.number_input("Rolling window (days, ending today)", min_value=1, max_value=365, value=45, step=1, key="rtk_roll_days")
    lo = today_ts - pd.Timedelta(days=int(rolling_days))
    hi = today_ts

    # Build first-enrollment map for every parent (lifetime min pay date)
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt_all"]  = pay_dt_all
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt_all"].notna()].groupby("_parent_norm")["_pay_dt_all"].min()

    # For each row in d0, find the parent's first enrollment date
    ref_first_enroll_dt = d0[parent_email_col].map(lambda e: first_pay_any.get(str(e).strip().lower(), pd.NaT))

    within_roll = ref_first_enroll_dt.between(lo, hi, inclusive="both")
    before_roll = ref_first_enroll_dt.notna() & (ref_first_enroll_dt < lo)

    # UI toggle: show counts for within ≤N vs >N (rolling)
    show_within = st.toggle(f"Show ≤{int(rolling_days)}d (to today). Off shows >{int(rolling_days)}d", value=True, key="rtk_toggle_within")
    bucket_label = f"≤{int(rolling_days)}d (to today)" if show_within else f">{int(rolling_days)}d (to today)"
    bucket_mask = within_roll if show_within else before_roll

    # ---- KPIs ----
    deals_matching = int((c_in & bucket_mask).sum())
    total_matching_in_window = int(c_in.sum())  # all same-email deals in window
    k1, k2 = st.columns(2)
    with k1: st.metric(f"Deals where Referrer == Parent ({bucket_label})", deals_matching)
    with k2: st.metric("All deals where Referrer == Parent (in window)", total_matching_in_window)

    # Optional: show both counts side-by-side
    k3, k4 = st.columns(2)
    with k3: st.metric(f"≤{int(rolling_days)}d (to today)", int((c_in & within_roll).sum()))
    with k4: st.metric(f">{int(rolling_days)}d (to today)", int((c_in & before_roll).sum()))

    # ---- Cohort/MTD nuance: if MTD, only consider events (like enrollment) when also in Create-date window.
    # For this specific tracker, counts are based on Create Date window by design.

    # ---- Table output ----
    with st.expander("Show matching rows"):
        cols = []
        for c in [ref_col, parent_email_col, _create, "Deal Name","Record ID","Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source", "Payment Received Date","Payment Date","Enrolment Date","Paid On"]:
            if c and c in d0.columns and c not in cols:
                cols.append(c)
        detail = d0.loc[c_in & bucket_mask, cols].copy()
        # For transparency add two helper columns
        detail["First Parent Enrollment Date (lifetime)"] = ref_first_enroll_dt.loc[detail.index]
        detail["Rolling bucket"] = np.where(within_roll.loc[detail.index], f"≤{int(rolling_days)}d (to today)",
                                            np.where(before_roll.loc[detail.index], f">{int(rolling_days)}d (to today)", "Other/NA"))
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Ref_Tracker details",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_ref_tracker_details.csv",
            mime="text/csv",
            key="dl_ref_tracker_details_csv"
        )



# --- Router: Marketing → Ref_Tracker ---
try:
    _master_for_rtk = master if 'master' in globals() else st.session_state.get('nav_master', '')
    _view_for_rtk = view if 'view' in globals() else st.session_state.get('nav_sub', '')
    if _master_for_rtk == "Marketing" and _view_for_rtk == "Ref_Tracker":
        _render_marketing_ref_tracker(
            df_f=df_f,
            create_col=create_col,
            pay_col=pay_col,
        )
except Exception as _e_rtk:
    import streamlit as st
    st.error(f"Ref_Tracker failed: {type(_e_rtk).__name__}: {_e_rtk}")



def _render_marketing_ref_tracker(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
):
    """
    Marketing → Ref_Tracker (exact 2-bucket split)
    - Scope: deals in selected window where Deal referred by(Email) == Parent Email (normalized)
    - Split (rolling relative to today, default N=45):
        1) Within ≤N days: parent's first enrollment ∈ [today-N, today]
        2) Beyond N days: everything else (before <today-N, after today, or never enrolled)
    - Guarantee: within + beyond == total in-window
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Ref_Tracker")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    # ---- Resolve required columns ----
    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_col or not parent_email_col:
        st.error("Required email columns not found. Need both 'Deal referred by (Email)' and 'Parent Email'."); return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])

    if not _create:
        st.error("Create Date column not found."); return

    # ---- Mode & Date presets ----
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rtk_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rtk_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rtk_start")
        with c2: end_d   = st.date_input("End", value=today, key="rtk_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Normalize bounds
    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)
    today_ts = pd.Timestamp(today).normalize().tz_localize(None)

    # ---- Helpers ----
    def _norm_email(x):
        if pd.isna(x): return ""
        try: return str(x).strip().lower()
        except Exception: return ""

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try: dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception: dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try: dt = dt.dt.tz_localize(None)
            except Exception: pass
        return dt.dt.floor("D")

    # ---- Normalize & restrict to rows where ref==parent ----
    ref_norm_all    = df_f[ref_col].map(_norm_email)
    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    same_mask_all = (ref_norm_all != "") & (ref_norm_all == parent_norm_all)
    d0 = df_f.loc[same_mask_all].copy()

    # ---- Parse dates for filtered rows ----
    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)

    # ---- Deal window mask (counts are based on Create Date window) ----
    c_in = create_dt.between(start_ts, end_ts)

    # ---- Build lifetime first-enrollment map for all parents ----
    pay_dt_full = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt"]  = pay_dt_full
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt"].notna()].groupby("_parent_norm")["_pay_dt"].min()

    # For each row in d0, get parent's first enrollment date
    parent_first_enroll = d0[parent_email_col].map(lambda e: first_pay_any.get(str(e).strip().lower(), pd.NaT))

    # ---- Exact 2-bucket split relative to TODAY (rolling) ----
    N = st.number_input("Rolling window N days (relative to today)", min_value=1, max_value=365, value=45, step=1, key="rtk_N")
    lo = today_ts - pd.Timedelta(days=int(N))
    hi = today_ts  # inclusive

    within_bucket_mask = parent_first_enroll.between(lo, hi, inclusive="both")
    # "Beyond" explicitly includes: earlier than lo, after today, and NaT (never enrolled)
    beyond_bucket_mask = ~within_bucket_mask  # exact complement

    # Restrict to in-window deals
    within_count = int((c_in & within_bucket_mask).sum())
    beyond_count = int((c_in & beyond_bucket_mask).sum())
    total_count  = int(c_in.sum())

    # Safety guard: enforce equality in display (should match by construction)
    if within_count + beyond_count != total_count:
        st.warning(f"Partition mismatch detected: within({within_count}) + beyond({beyond_count}) != total({total_count}). Displaying computed totals anyway.")

    # ---- KPIs ----
    k1, k2, k3 = st.columns(3)
    with k1: st.metric(f"Within ≤{int(N)}d (to today)", within_count)
    with k2: st.metric(f"Beyond {int(N)}d (to today)", beyond_count)
    with k3: st.metric("Total (in window)", total_count)

    # ---- Bucket selector for table ----
    bucket_choice = st.radio("Show rows for:", [f"Within ≤{int(N)}d", f"Beyond {int(N)}d", "Both"], index=2, horizontal=True, key="rtk_bucket_choice")
    if bucket_choice.startswith("Within"):
        row_mask = c_in & within_bucket_mask
    elif bucket_choice.startswith("Beyond"):
        row_mask = c_in & beyond_bucket_mask
    else:
        row_mask = c_in  # both

    # ---- Details table ----
    with st.expander("Show rows"):
        cols = []
        for c in [ref_col, parent_email_col, _create, "Deal Name","Record ID","Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source", "Payment Received Date","Payment Date","Enrolment Date","Paid On"]:
            if c and c in d0.columns and c not in cols:
                cols.append(c)
        detail = d0.loc[row_mask, cols].copy()
        # Annotate bucket + first enrollment
        detail["First Parent Enrollment Date (lifetime)"] = parent_first_enroll.loc[detail.index]
        detail["Bucket (rolling vs today)"] = np.where(within_bucket_mask.loc[detail.index], f"Within ≤{int(N)}d", f"Beyond {int(N)}d")
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Ref_Tracker rows",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_ref_tracker_rows.csv",
            mime="text/csv",
            key="dl_ref_tracker_rows_csv"
        )



def _render_marketing_ref_tracker(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
):
    """
    Marketing → Ref_Tracker (DEAL-RELATIVE 45d window)
    - Scope: referral deals in selected window where Referrer Email == Parent Email (normalized)
    - For each deal, compute Δdays = (Deal Create Date − Referrer's FIRST Enrollment Date)
    - Buckets (exact 2-way split so A+B == Total):
        A) Within ≤N days: 0 ≤ Δdays ≤ N  (parent enrolled before the deal, within N days)
        B) Beyond N days: Δdays < 0 (parent enrolled after the deal) OR Δdays > N OR no enrollment
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Ref_Tracker")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    # ---- Resolve required columns ----
    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_col or not parent_email_col:
        st.error("Required email columns not found. Need both 'Deal referred by (Email)' and 'Parent Email'."); return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])

    if not _create:
        st.error("Create Date column not found."); return

    # ---- Mode & Date presets ----
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rtk_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rtk_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rtk_start")
        with c2: end_d   = st.date_input("End", value=today, key="rtk_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Normalize bounds
    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)

    # ---- Helpers ----
    def _norm_email(x):
        if pd.isna(x): return ""
        try: return str(x).strip().lower()
        except Exception: return ""

    def _to_day_ts(s: pd.Series) -> pd.Series:
        try: dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception: dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try: dt = dt.dt.tz_localize(None)
            except Exception: pass
        return dt.dt.floor("D")

    # ---- Normalize & restrict to rows where ref==parent ----
    ref_norm_all    = df_f[ref_col].map(_norm_email)
    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    same_mask_all = (ref_norm_all != "") & (ref_norm_all == parent_norm_all)
    d0 = df_f.loc[same_mask_all].copy()

    # ---- Parse dates for filtered rows ----
    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)

    # ---- Deal window mask (counts based on Create Date window) ----
    c_in = create_dt.between(start_ts, end_ts)

    # ---- Build lifetime FIRST enrollment map for parents ----
    pay_dt_full = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt"]  = pay_dt_full
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt"].notna()].groupby("_parent_norm")["_pay_dt"].min()

    # Map each row's parent to their first enrollment date
    parent_first_enroll = d0[parent_email_col].map(lambda e: first_pay_any.get(str(e).strip().lower(), pd.NaT))

    # ---- DEAL-RELATIVE 2-bucket split: Δdays = create - first_enroll ----
    N = st.number_input("Window N days relative to deal date", min_value=1, max_value=365, value=45, step=1, key="rtk_N_deal")
    delta_days = (create_dt - parent_first_enroll).dt.days  # NaN if parent_first_enroll is NaT

    within_mask = delta_days.between(0, int(N))  # 0 ≤ Δ ≤ N
    # Everything else is Beyond (Δ<0, Δ>N, NaN)
    beyond_mask = ~within_mask

    within_count = int((c_in & within_mask).sum())
    beyond_count = int((c_in & beyond_mask).sum())
    total_count  = int(c_in.sum())

    # ---- KPIs ----
    k1, k2, k3 = st.columns(3)
    with k1: st.metric(f"A: Within ≤{int(N)}d of deal", within_count)
    with k2: st.metric(f"B: Beyond {int(N)}d of deal", beyond_count)
    with k3: st.metric("Total referral deals in window", total_count)
    
    
    # ---- Pie chart: Within / Beyond / Remaining ----
    remaining = max(total_count - (within_count + beyond_count), 0)
    pie_df = pd.DataFrame({
        "Bucket": [f"Within ≤{int(N)}d", f"Beyond {int(N)}d", "Remaining"],
        "Count": [within_count, beyond_count, remaining],
    })
    _pie_rendered = False

    # 1) Try Plotly
    try:
        import plotly.express as px
        fig = px.pie(pie_df, names="Bucket", values="Count", title="Referral deals split", hole=0.0)
        st.plotly_chart(fig, use_container_width=True)
        _pie_rendered = True
    except Exception:
        _pie_rendered = False

    # 2) Try Vega-Lite (built into Streamlit, no extra deps)
    if not _pie_rendered:
        try:
            spec = {
                "mark": {"type": "arc"},
                "encoding": {
                    "theta": {"field": "Count", "type": "quantitative"},
                    "color": {"field": "Bucket", "type": "nominal"},
                    "tooltip": [
                        {"field": "Bucket", "type": "nominal"},
                        {"field": "Count", "type": "quantitative"}
                    ]
                }
            }
            st.vega_lite_chart(pie_df, spec, use_container_width=True)
            _pie_rendered = True
        except Exception:
            _pie_rendered = False

    # 3) Try Matplotlib
    if not _pie_rendered:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.pie(pie_df["Count"], labels=pie_df["Bucket"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            ax.set_title("Referral deals split")
            st.pyplot(fig, use_container_width=True)
            _pie_rendered = True
        except Exception:
            _pie_rendered = False

    # 4) Fallback: show counts
    if not _pie_rendered:
        st.info("Chart libraries unavailable. Showing counts instead.")
        st.write({"Within": within_count, "Beyond": beyond_count, "Remaining": remaining})

    # ---- Table selector ----

    bucket_choice = st.radio("Show rows for:", [f"A: Within ≤{int(N)}d", f"B: Beyond {int(N)}d", "Both"], index=2, horizontal=True, key="rtk_bucket_choice_deal")
    if bucket_choice.startswith("A:"):
        row_mask = c_in & within_mask
    elif bucket_choice.startswith("B:"):
        row_mask = c_in & beyond_mask
    else:
        row_mask = c_in

    # ---- Details table ----
    with st.expander("Show rows"):
        cols = []
        for c in [ref_col, parent_email_col, _create, "Deal Name","Record ID","Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source", "Payment Received Date","Payment Date","Enrolment Date","Paid On"]:
            if c and c in d0.columns and c not in cols:
                cols.append(c)
        detail = d0.loc[row_mask, cols].copy()
        detail["First Parent Enrollment Date (lifetime)"] = parent_first_enroll.loc[detail.index]
        detail["Δdays (deal_create − first_enroll)"] = delta_days.loc[detail.index]
        detail["Bucket"] = np.where(within_mask.loc[detail.index], f"A: ≤{int(N)}d", f"B: >{int(N)}d / after / none")
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Ref_Tracker (deal-relative buckets)",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_ref_tracker_deal_relative.csv",
            mime="text/csv",
            key="dl_ref_tracker_deal_relative_csv"
        )



def _render_marketing_referral_split(
    df_f: pd.DataFrame,
    create_col: str | None,
    pay_col: str | None,
):
    """
    Marketing → Referral Split
    Focus: deals where Referral Intent Source == "Sales Generated" AND referrer email == parent email.
    For each deal in the selected window, compute Δdays = (Deal Create Date − FIRST Enrollment Date of the referrer parent).
    Buckets:
      A) Within ≤N days: 0 ≤ Δdays ≤ N
      B) Beyond N days: Δdays < 0 OR Δdays > N OR no enrollment found
    Guarantee: A + B == Total (in-window).
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta

    st.subheader("Marketing — Referral Split")

    if df_f is None or getattr(df_f, "empty", True):
        st.info("No data available."); return

    # --- Resolve columns ---
    ref_intent_col = find_col(df_f, ["Referral Intent Source","Referral_Intent_Source","Referral Intent source","Referral intent source"])
    ref_col = find_col(df_f, [
        "Deal referred by(Email):","Deal referred by(Email)",
        "Deal referred by (Email)","Deal referred by Email",
        "Referred By Email","Referral Email","Referrer Email"
    ])
    parent_email_col = find_col(df_f, ["Parent Email","Parent email","Email","Parent_Email"])

    if not ref_intent_col:
        st.error("Column for 'Referral Intent Source' not found."); return
    if not ref_col or not parent_email_col:
        st.error("Need both 'Deal referred by (Email)' and 'Parent Email' columns."); return

    _create = create_col if (create_col and create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
    _pay    = pay_col    if (pay_col and pay_col in df_f.columns)       else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])
    if not _create:
        st.error("Create Date column not found."); return

    # --- Mode & date presets ---
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="rfs_mode")

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True, key="rfs_rng")

    def _month_bounds(d: date):
        from calendar import monthrange
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = _month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="rfs_start")
        with c2: end_d   = st.date_input("End", value=today, key="rfs_end")
        if end_d < start_d: start_d, end_d = end_d, start_d

    # Normalize bounds
    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)

    # --- Helpers ---
    def _norm_email(x):
        if pd.isna(x): return ""
        try: return str(x).strip().lower()
        except Exception: return ""
    def _to_day_ts(s: pd.Series) -> pd.Series:
        try: dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception: dt = pd.to_datetime(s, errors="coerce")
        try:
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try: dt = dt.dt.tz_localize(None)
            except Exception: pass
        return dt.dt.floor("D")

    # --- Filter to target deals ---
    ref_norm_all    = df_f[ref_col].map(_norm_email)
    parent_norm_all = df_f[parent_email_col].map(_norm_email)
    same_email_all  = (ref_norm_all != "") & (ref_norm_all == parent_norm_all)

    is_sales_gen = df_f[ref_intent_col].astype(str).str.strip().str.lower() == "sales generated"
    base_mask = same_email_all & is_sales_gen
    d0 = df_f.loc[base_mask].copy()

    if d0.empty:
        st.info("No rows where Referral Intent Source = 'Sales Generated' AND referrer email matches parent email in the selected data.")
        return

    # --- Dates for filtered deals ---
    create_dt = _to_day_ts(d0[_create]) if _create in d0.columns else pd.Series(pd.NaT, index=d0.index)
    c_in = create_dt.between(start_ts, end_ts)

    # --- Parent's FIRST enrollment map (lifetime) ---
    pay_dt_full = _to_day_ts(df_f[_pay]) if (_pay and _pay in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    g_all = df_f.copy()
    g_all["_parent_norm"] = parent_norm_all
    g_all["_pay_dt"] = pay_dt_full
    first_pay_any = g_all.loc[(g_all["_parent_norm"] != "") & g_all["_pay_dt"].notna()].groupby("_parent_norm")["_pay_dt"].min()

    parent_first_enroll = d0[parent_email_col].map(lambda e: first_pay_any.get(str(e).strip().lower(), pd.NaT))

    # --- Deal-relative A/B split ---
    N = st.number_input("Window N days relative to deal date", min_value=1, max_value=365, value=45, step=1, key="rfs_N")
    delta_days = (create_dt - parent_first_enroll).dt.days
    within_mask = delta_days.between(0, int(N))          # 0 ≤ Δ ≤ N
    beyond_mask = ~within_mask                           # everything else

    A = int((c_in & within_mask).sum())
    B = int((c_in & beyond_mask).sum())
    Total = int(c_in.sum())

    # --- KPIs ---
    k1, k2, k3 = st.columns(3)
    with k1: st.metric(f"A: Within ≤{int(N)}d of deal", A)
    with k2: st.metric(f"B: Beyond {int(N)}d of deal", B)
    with k3: st.metric("Total (Sales Generated referrals)", Total)

    if A + B != Total:
        st.warning(f"Partition mismatch: A({A}) + B({B}) != Total({Total}). Please review date columns.")

    # --- (Optional) simple bar instead of pie for clarity in this view ---
    try:
        import altair as alt
        bar_df = pd.DataFrame({"Bucket":[f"A ≤{int(N)}d", f"B >{int(N)}d"], "Count":[A,B]})
        chart = alt.Chart(bar_df).mark_bar().encode(x="Bucket:N", y="Count:Q", tooltip=["Bucket","Count"]).properties(title="Sales Generated referral split")
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass

    # --- Details table ---
    bucket_choice = st.radio("Show rows for:", [f"A: Within ≤{int(N)}d", f"B: Beyond {int(N)}d", "Both"], index=2, horizontal=True, key="rfs_choice")
    if bucket_choice.startswith("A:"):
        row_mask = c_in & within_mask
    elif bucket_choice.startswith("B:"):
        row_mask = c_in & beyond_mask
    else:
        row_mask = c_in

    with st.expander("Show rows"):
        cols = []
        for c in [ref_intent_col, ref_col, parent_email_col, _create, "Deal Name","Record ID",
                  "Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source",
                  "Payment Received Date","Payment Date","Enrolment Date","Paid On"]:
            if c and c in d0.columns and c not in cols:
                cols.append(c)
        detail = d0.loc[row_mask, cols].copy()
        detail["First Parent Enrollment Date (lifetime)"] = parent_first_enroll.loc[detail.index]
        detail["Δdays (deal_create − first_enroll)"] = delta_days.loc[detail.index]
        detail["Bucket"] = np.where(within_mask.loc[detail.index], f"A: ≤{int(N)}d", f"B: >{int(N)}d / after / none")
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Referral Split rows",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_split_rows.csv",
            mime="text/csv",
            key="dl_referral_split_rows_csv"
        )



# --- Router: Marketing → Referral Split ---
try:
    _master_for_rfs = master if 'master' in globals() else st.session_state.get('nav_master', '')
    _view_for_rfs = view if 'view' in globals() else st.session_state.get('nav_sub', '')
    if _master_for_rfs == "Marketing" and _view_for_rfs == "Referral Split":
        _render_marketing_referral_split(
            df_f=df_f,
            create_col=create_col,
            pay_col=pay_col,
        )
except Exception as _e_rfs:
    import streamlit as st
    st.error(f"Referral Split failed: {type(_e_rfs).__name__}: {_e_rfs}")



# ============================
# BEGIN: Marketing -> Talk Time (appended by patch)
# ============================
import hashlib as _tt_hashlib
import pandas as _tt_pd
import numpy as _tt_np
import streamlit as _tt_st
import altair as _tt_alt

def tt__find_col(df, candidates, default=None):
    cols = list(df.columns)
    # case-insensitive match
    lower_map = {c.lower().strip(): c for c in cols}
    for name in candidates or []:
        key = str(name).lower().strip()
        if key in lower_map: return lower_map[key]
    # contains heuristic
    def norm(s): 
        import re as _re
        return _re.sub(r'[^a-z0-9]+','', str(s).lower())
    cand_norm = [norm(x) for x in (candidates or [])]
    for c in cols:
        nc = norm(c)
        for cn in cand_norm:
            if cn and cn in nc: 
                return c
    return default
from datetime import datetime as _tt_dt, date as _tt_date, time as _tt_time, timedelta as _tt_timedelta
import re

_TT_TZ = "Asia/Kolkata"

def tt__read_csv_flexible(upfile, manual_sep=None):
    """Robust CSV loader for messy exports.
    - Tries encoding: utf-8-sig, utf-8, latin-1
    - Tries sep: manual -> auto (engine='python', sep=None), then [',',';','\\t','|']
    - Skips malformed lines with on_bad_lines='skip'
    Returns (df, debug_log_str)
    """
    import io
    attempts = []
    raw = upfile.read()
    def try_read(enc, sep, infer, engine, bad):
        bio = io.BytesIO(raw)
        kw = dict(engine=engine, on_bad_lines=bad)
        if enc: kw["encoding"] = enc
        if infer:
            kw["sep"] = None
        else:
            if sep is not None: kw["sep"] = sep
        try:
            df = _tt_pd.read_csv(bio, **kw)
            return df, f"OK enc={enc} sep={'AUTO' if infer else repr(sep)} engine={engine} bad={bad}"
        except Exception as e:
            return None, f"FAIL enc={enc} sep={'AUTO' if infer else repr(sep)} engine={engine} bad={bad} -> {type(e).__name__}: {e}"
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    seps = [",",";","\\t","|"]
    if manual_sep is not None:
        for enc in encodings:
            for bad in ["skip"]:
                df, note = try_read(enc, manual_sep, False, "python", bad)
                attempts.append(note)
                if df is not None:
                    return df, "\\n".join(attempts)
    for enc in encodings:
        for bad in ["skip"]:
            df, note = try_read(enc, None, True, "python", bad)
            attempts.append(note)
            if df is not None:
                return df, "\\n".join(attempts)
    for enc in encodings:
        for sep in seps:
            for bad in ["skip"]:
                df, note = try_read(enc, sep, False, "python", bad)
                attempts.append(note)
                if df is not None:
                    return df, "\\n".join(attempts)
    return None, "\\n".join(attempts)

def tt__duration_to_secs(s):
    """Parse Call Duration into seconds. Accepts 'HH:MM:SS', 'MM:SS', 'SS', '1h 2m 3s', '0:00:45.000', '1.02.03' and more."""
    if s is None or (isinstance(s, float) and _tt_np.isnan(s)):
        return _tt_np.nan
    if isinstance(s, (int, float)) and not _tt_np.isnan(s):
        return int(round(float(s)))
    ss = str(s).strip()
    if ss == "" or ss.lower() in {"nan", "none", "-"}:
        return _tt_np.nan
    ss = ss.replace(";", ":").replace(".", ":")
    m = re.findall(r'(\\d+)\\s*([hms])', ss.lower())
    if m:
        hh = mm = sec = 0
        for val, unit in m:
            if unit == 'h': hh = int(val)
            elif unit == 'm': mm = int(val)
            elif unit == 's': sec = int(val)
        return hh*3600 + mm*60 + sec
    if re.fullmatch(r'\\d+', ss):
        return int(ss)
    parts = [p for p in ss.split(":") if p != ""]
    try:
        parts = [int(float(p)) for p in parts]
    except:
        return _tt_np.nan
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        mm, sec = parts
        return mm*60 + sec
    if len(parts) >= 3:
        hh, mm, sec = (parts + [0,0,0])[:3]
        return hh*3600 + mm*60 + sec

def tt__excel_serial_to_date(val):
    try:
        f = float(val)
    except:
        return _tt_pd.NaT
    if f <= 0 or f > 100000:
        return _tt_pd.NaT
    try:
        return (_tt_pd.to_datetime("1899-12-30") + _tt_pd.to_timedelta(f, unit="D")).date()
    except Exception:
        return _tt_pd.NaT

def tt__parse_date(d):
    if isinstance(d, _tt_pd.Timestamp):
        return d.date()
    if isinstance(d, (int, float)) and not _tt_pd.isna(d):
        return tt__excel_serial_to_date(d)
    s = str(d).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return _tt_pd.NaT
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y"):
        try:
            return _tt_pd.to_datetime(s, format=fmt, errors="raise").date()
        except Exception:
            pass
    dt = _tt_pd.to_datetime(s, errors="coerce", dayfirst=True)
    if _tt_pd.isna(dt):
        dt = _tt_pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt.date() if not _tt_pd.isna(dt) else _tt_pd.NaT

def tt__parse_time(t):
    if isinstance(t, (int, float)) and not _tt_pd.isna(t):
        secs = int(round(float(t))); secs = max(0, min(secs, 24*3600-1))
        hh = secs//3600; mm=(secs%3600)//60; ss=secs%60
        try:
            return _tt_pd.Timestamp(year=2000, month=1, day=1, hour=hh, minute=mm, second=ss).time()
        except:
            return _tt_pd.NaT
    s = str(t).strip()
    if not s or s.lower() in {"nan","none","-"}: return _tt_pd.NaT
    sn = s.replace(".", ":").replace(";", ":")
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return _tt_pd.to_datetime(sn, format=fmt, errors="raise").time()
        except Exception:
            pass
    dt = _tt_pd.to_datetime(sn, errors="coerce")
    return dt.time() if not _tt_pd.isna(dt) else _tt_pd.NaT

def tt__combine_dt(row):
    d = row.get("_tt_Date"); t = row.get("_tt_Time")
    if _tt_pd.isna(d) and _tt_pd.isna(t): return _tt_pd.NaT
    if _tt_pd.isna(d) or _tt_pd.isna(t):
        raw_d = row.get("Date"); raw_t = row.get("Time")
        combo = f"{raw_d} {raw_t}".strip()
        dt = _tt_pd.to_datetime(combo, errors="coerce", dayfirst=True)
        if _tt_pd.isna(dt):
            dt = _tt_pd.to_datetime(combo, errors="coerce", dayfirst=False)
        return dt if not _tt_pd.isna(dt) else _tt_pd.NaT
    try:
        return _tt_pd.Timestamp.combine(d, t)
    except Exception:
        return _tt_pd.NaT

def tt__fmt_hms(total_seconds):
    if _tt_pd.isna(total_seconds): return "00:00:00"
    total_seconds = int(total_seconds)
    h = total_seconds // 3600; m = (total_seconds % 3600) // 60; s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def tt__date_preset_bounds(preset, today=None):
    tz_today = _tt_pd.Timestamp.now(tz="Asia/Kolkata").date() if today is None else today
    if preset == "Today": return tz_today, tz_today
    if preset == "Yesterday": y = tz_today - _tt_timedelta(days=1); return y, y
    if preset == "This Month": start = tz_today.replace(day=1); return start, tz_today
    if preset == "Last Month":
        first_this = tz_today.replace(day=1); last_month_end = first_this - _tt_timedelta(days=1)
        start = last_month_end.replace(day=1); return start, last_month_end
    return tz_today, tz_today

def _render_marketing_talk_time(df):
    _tt_st.subheader("Talk Time")
    _tt_st.caption("Upload call activity CSV and analyze talk time by agent/country with 24h patterns.")
    up = _tt_st.file_uploader("Upload activity feed CSV", type=["csv"], key="tt_uploader")
    if up is not None:
        manual_sep = _tt_st.selectbox("Delimiter", ["Auto-detect","Comma ,","Semicolon ;","Tab \\t","Pipe |"], index=0, key="tt_sep")
        sep_map = {"Auto-detect": None, "Comma ,": ",", "Semicolon ;": ";", "Tab \\t": "\\t", "Pipe |": "|"}
        df_try, debug_log = tt__read_csv_flexible(up, manual_sep=sep_map[manual_sep])
        if df_try is None:
            _tt_st.error("Unable to read CSV after multiple attempts. Please try a different delimiter or encoding.")
            _tt_st.expander("Open parser debug log").write(debug_log); return
        dff = df_try
    else:
        dff = df.copy() if df is not None else None
        if dff is None or not set(["Date","Time","Caller","Call Type","Country Name","Call Status","Call Duration"]).issubset(set(map(str, dff.columns))):
            _tt_st.info("Please upload the **activity feed CSV** with columns: Date, Time, Caller, Call Type, Country Name, Call Status, Call Duration."); return

    # Map your CSV columns to canonical names used below
    _col_date    = tt__find_col(dff, ["Date","Call Date","Start Date"])
    _col_time    = tt__find_col(dff, ["Client Answer Time","Call Time","Answer Time","Time","Start Time"])
    _col_agent   = tt__find_col(dff, ["Email","Caller","Agent","User","Counsellor","Counselor","Advisor"])
    _col_country = tt__find_col(dff, ["Country Name","Country"])
    _col_status  = tt__find_col(dff, ["Call Status","Status"])
    _col_type    = tt__find_col(dff, ["Call Type","Type"])
    _col_dur     = tt__find_col(dff, ["Call Duration","Duration","Talk Time"])

    # UI fallbacks if any key column missing
    if _col_agent is None:
        _tt_st.warning("Could not detect the Agent column. Please choose it:")
        _col_agent = _tt_st.selectbox("Agent column", options=dff.columns.tolist(), key="tt_pick_agent")
    if _col_date is None:
        _tt_st.warning("Could not detect the Date column. Please choose it:")
        _col_date = _tt_st.selectbox("Date column", options=dff.columns.tolist(), key="tt_pick_date")
    if _col_time is None:
        _tt_st.warning("Could not detect the Time column. Please choose it:")
        _col_time = _tt_st.selectbox("Time column", options=dff.columns.tolist(), key="tt_pick_time")
    if _col_dur is None:
        _tt_st.warning("Could not detect the Call Duration column. Please choose it:")
        _col_dur = _tt_st.selectbox("Duration column", options=dff.columns.tolist(), key="tt_pick_dur")
    if _col_status is None and "Call Status" in dff.columns:
        _col_status = "Call Status"

    # Build a working frame with canonical labels expected by the rest of the logic
    remap = {
        "Date": _col_date,
        "Time": _col_time,
        "Caller": _col_agent,
        "Call Type": _col_type,
        "Country Name": _col_country,
        "Call Status": _col_status,
        "Call Duration": _col_dur
    }
    use_cols = [v for v in remap.values() if v is not None]
    dff = dff[use_cols].copy()
    # Rename to canonical if present
    inv = {v:k for k,v in remap.items() if v is not None}
    dff = dff.rename(columns=inv)


    # Parse
    dff["_tt_Date"] = dff["Date"].apply(tt__parse_date) if "Date" in dff.columns else _tt_pd.NaT
    dff["_tt_Time"] = dff["Time"].apply(tt__parse_time) if "Time" in dff.columns else _tt_pd.NaT
    combo = (dff["Date"].astype(str).str.strip() + " " + dff["Time"].astype(str).str.strip()).str.strip()
    dt_combo = _tt_pd.to_datetime(combo, errors="coerce", dayfirst=True)
    dt_combo2 = _tt_pd.to_datetime(combo, errors="coerce", dayfirst=False)
    dff["_tt_dt"] = _tt_pd.NaT
    both_ok = (~dff["_tt_Date"].isna()) & (~dff["_tt_Time"].isna())
    dff.loc[both_ok, "_tt_dt"] = dff.loc[both_ok].apply(tt__combine_dt, axis=1)
    remaining = dff["_tt_dt"].isna()
    dff.loc[remaining, "_tt_dt"] = dt_combo[remaining]
    remaining = dff["_tt_dt"].isna()
    dff.loc[remaining, "_tt_dt"] = dt_combo2[remaining]
    dff["_tt_secs"] = dff["Call Duration"].apply(tt__duration_to_secs) if "Call Duration" in dff.columns else _tt_np.nan
    dff["_tt_hour"] = dff["_tt_dt"].dt.hour
    dff["_tt_is_60"] = dff["_tt_secs"] > 60

    dff["_reason_dt"] = _tt_np.where(dff["_tt_dt"].isna(), "Bad Date/Time", "")
    dff["_reason_secs"] = _tt_np.where(dff["_tt_secs"].isna(), "Bad Duration", "")
    dff["_bad"] = dff["_tt_dt"].isna() | dff["_tt_secs"].isna()
    bad_rows = dff["_bad"]
    _bad_count = int(bad_rows.sum())
    if _bad_count:
        _tt_st.warning(f"Excluded {_bad_count} rows with unparseable Date/Time or Duration.")
        cols_excl = [c for c in ["Date","Time","Caller","Call Type","Country Name","Call Status","Call Duration","_reason_dt","_reason_secs"] if (c in dff.columns) or c.startswith("_reason")]
        excl = dff.loc[bad_rows, cols_excl].copy()
        _tt_st.download_button("Download excluded rows (CSV)", excl.to_csv(index=False).encode("utf-8"), "talk_time_excluded_rows.csv", "text/csv", key="tt_dl_excluded")

    dff = dff[~bad_rows].copy()

    _tt_st.markdown("### Filters")
    c1, c2, c3 = _tt_st.columns([1,1,1])
    with c1:
        preset = _tt_st.selectbox("Date preset", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, key="tt_preset")
    with c2:
        today = _tt_pd.Timestamp.now(tz="Asia/Kolkata").date()
        start_default, end_default = tt__date_preset_bounds(preset, today=today)
        start_date = _tt_st.date_input("Start date", value=start_default, key="tt_start")
    with c3:
        end_date = _tt_st.date_input("End date", value=end_default, key="tt_end")
    if start_date > end_date: start_date, end_date = end_date, start_date

    c4, c5, c6, c7 = _tt_st.columns([1,1,1,1])
    agents = sorted(dff["Caller"].dropna().unique().tolist()) if "Caller" in dff.columns else []
    countries = sorted(dff["Country Name"].dropna().unique().tolist()) if "Country Name" in dff.columns else []
    statuses = sorted(dff["Call Status"].dropna().unique().tolist()) if "Call Status" in dff.columns else []
    ctypes = sorted(dff["Call Type"].dropna().unique().tolist()) if "Call Type" in dff.columns else []
    with c4: sel_agents = _tt_st.multiselect("Agent(s)", agents, default=agents[:10], key="tt_agents")
    with c5: sel_countries = _tt_st.multiselect("Country", countries, default=countries[:10], key="tt_countries")
    with c6: status_mode = _tt_st.selectbox("Status", ["Connected only","All statuses"], index=0, key="tt_status_mode")
    with c7: sel_ctype = _tt_st.multiselect("Call Type", ctypes, default=ctypes, key="tt_ctype")
    gate_mode = _tt_st.radio("Duration", ["All calls", "> 60s only"], index=0, key="tt_gate")

    mask = (dff["_tt_dt"].dt.date.between(start_date, end_date))
    if sel_agents:   mask &= dff["Caller"].isin(sel_agents)
    if countries and sel_countries: mask &= dff["Country Name"].isin(sel_countries)
    if ctypes and sel_ctype:        mask &= dff["Call Type"].isin(sel_ctype)
    if status_mode == "Connected only" and "Call Status" in dff.columns:
        mask &= (dff["_tt_secs"] > 0)

    dfv = dff[mask].copy()
    if gate_mode == "> 60s only": dfv = dfv[dfv["_tt_is_60"]].copy()

    total_secs = dfv["_tt_secs"].sum() if len(dfv) else 0
    n_calls = int(len(dfv)); avg_secs = (total_secs / n_calls) if n_calls > 0 else 0; pct_60 = (100.0 * dfv["_tt_is_60"].mean()) if len(dfv) else 0.0
    k1,k2,k3,k4 = _tt_st.columns(4)
    k1.metric("Total Talk Time", tt__fmt_hms(total_secs))
    k2.metric("# Calls", f"{n_calls:,}")
    k3.metric("Avg Talk / Call", tt__fmt_hms(int(avg_secs)))
    k4.metric("% Calls > 60s", f"{pct_60:.1f}%")

    _tt_st.markdown("### Agent-wise Talk Time")
    if len(dfv):
        g_agent = dfv.groupby("Caller", dropna=True)["_tt_secs"].agg(["count", "sum", "mean"]).reset_index()
        g_agent = g_agent.rename(columns={"Caller":"Agent","count":"Calls","sum":"Total Seconds","mean":"Avg Seconds"})
        g_agent["Calls >60s"] = dfv.groupby("Caller")["_tt_is_60"].sum().reindex(g_agent["Agent"]).fillna(0).astype(int).values
        g_agent = g_agent.sort_values("Total Seconds", ascending=False)
        g_agent["Total Talk"] = g_agent["Total Seconds"].apply(lambda x: tt__fmt_hms(int(x)))
        g_agent["Avg Talk"] = g_agent["Avg Seconds"].apply(lambda x: tt__fmt_hms(int(x)))
        g_show = g_agent[["Agent","Calls","Calls >60s","Total Talk","Avg Talk","Total Seconds"]]
        _tt_st.dataframe(g_show, use_container_width=True)
        _tt_st.download_button("Download Agent Totals (CSV)", g_show.to_csv(index=False).encode("utf-8"), "agent_talk_time.csv", "text/csv", key="tt_dl_agent")
    else:
        _tt_st.info("No rows after filters.")

    if "Country Name" in dfv.columns and dfv["Country Name"].notna().any():
        _tt_st.markdown("### Country-wise Talk Time")
        g_country = dfv.groupby("Country Name", dropna=True)["_tt_secs"].agg(["count", "sum", "mean"]).reset_index()
        g_country = g_country.rename(columns={"Country Name":"Country","count":"Calls","sum":"Total Seconds","mean":"Avg Seconds"})
        g_country["Calls >60s"] = dfv.groupby("Country Name")["_tt_is_60"].sum().reindex(g_country["Country"]).fillna(0).astype(int).values
        g_country = g_country.sort_values("Total Seconds", ascending=False)
        g_country["Total Talk"] = g_country["Total Seconds"].apply(lambda x: tt__fmt_hms(int(x)))
        g_country["Avg Talk"] = g_country["Avg Seconds"].apply(lambda x: tt__fmt_hms(int(x)))
        g2_show = g_country[["Country","Calls","Calls >60s","Total Talk","Avg Talk","Total Seconds"]]
        _tt_st.dataframe(g2_show, use_container_width=True)
        _tt_st.download_button("Download Country Totals (CSV)", g2_show.to_csv(index=False).encode("utf-8"), "country_talk_time.csv", "text/csv", key="tt_dl_country")

    _tt_st.markdown("### 24h Calling Pattern (Bubble)")
    group_by = _tt_st.radio("Group by", ["Agent","Country"], index=0, horizontal=True, key="tt_group_by")
    if group_by == "Agent": gb_col = "Caller"; y_title = "Agent"
    else: gb_col = "Country Name"; y_title = "Country"

    if gb_col not in dfv.columns or dfv[gb_col].isna().all():
        _tt_st.info(f"{y_title} column not present/empty; showing hour totals only.")
        chs = dfv.groupby("_tt_hour")["_tt_secs"].sum().reset_index().rename(columns={"_tt_hour":"Hour","_tt_secs":"Total Seconds"})
        base = _tt_alt.Chart(chs).mark_circle().encode(
            x=_tt_alt.X("Hour:O", title="Hour (0–23)"),
            y=_tt_alt.Y("Hour:O", title=y_title),
            size=_tt_alt.Size("Total Seconds:Q", legend=None),
            tooltip=["Hour","Total Seconds"]
        ).interactive()
        _tt_st.altair_chart(base, use_container_width=True)
    else:
        csrc_all = dfv.groupby([gb_col, "_tt_hour"])["_tt_secs"].sum().reset_index().rename(columns={gb_col: y_title, "_tt_hour":"Hour", "_tt_secs":"Total Seconds"})
        ch_all = _tt_alt.Chart(csrc_all).mark_circle().encode(
            x=_tt_alt.X("Hour:O", title="Hour (0–23)"),
            y=_tt_alt.Y(f"{y_title}:N", sort="-x"),
            size=_tt_alt.Size("Total Seconds:Q", legend=None),
            tooltip=[y_title, "Hour", _tt_alt.Tooltip("Total Seconds:Q", title="Total Seconds")]
        ).properties(title="All Calls").interactive()
        _tt_st.altair_chart(ch_all, use_container_width=True)

        dfv60 = dfv[dfv["_tt_is_60"]].copy()
        csrc_60 = dfv60.groupby([gb_col, "_tt_hour"])["_tt_secs"].sum().reset_index().rename(columns={gb_col: y_title, "_tt_hour":"Hour", "_tt_secs":"Total Seconds"})
        ch_60 = _tt_alt.Chart(csrc_60).mark_circle().encode(
            x=_tt_alt.X("Hour:O", title="Hour (0–23)"),
            y=_tt_alt.Y(f"{y_title}:N", sort="-x"),
            size=_tt_alt.Size("Total Seconds:Q", legend=None),
            tooltip=[y_title, "Hour", _tt_alt.Tooltip("Total Seconds:Q", title="Total Seconds")]
        ).properties(title="> 60s Calls Only").interactive()
        _tt_st.altair_chart(ch_60, use_container_width=True)
# ============================
# END: Marketing -> Talk Time (appended by patch)
# ============================



# ============================
# BEGIN: Marketing -> Referral Box (appended by patch)
# ============================
import pandas as _rb_pd
import numpy as _rb_np
import streamlit as _rb_st
from datetime import date as _rb_date, timedelta as _rb_timedelta
import re

def _rb_date_preset_bounds(preset, today=None):
    tz_today = _rb_pd.Timestamp.now(tz="Asia/Kolkata").date() if today is None else today
    if preset == "Today": return tz_today, tz_today
    if preset == "Yesterday": y = tz_today - _rb_timedelta(days=1); return y, y
    if preset == "This Month": start = tz_today.replace(day=1); return start, tz_today
    if preset == "Last Month":
        first_this = tz_today.replace(day=1)
        last_month_end = first_this - _rb_timedelta(days=1)
        start = last_month_end.replace(day=1)
        return start, last_month_end
    return tz_today, tz_today

def _rb_find_col(df, candidates, default=None, keywords=None):
    """
    Flexible column finder:
    - Exact case-insensitive match on any name in candidates
    - Normalized token match (remove non-alnum, collapse spaces/underscores)
    - Optional keyword ALL-match (every keyword must appear in normalized name)
    """
    cols = list(df.columns)
    norm = lambda s: re.sub(r'[^a-z0-9]+', '', str(s).lower())
    # 1) direct case-insensitive
    lower_map = {c.lower().strip(): c for c in cols}
    for name in candidates or []:
        key = str(name).lower().strip()
        if key in lower_map: return lower_map[key]
    # 2) normalized equality
    cand_norm = {norm(name): name for name in (candidates or [])}
    for c in cols:
        if norm(c) in cand_norm: return c
    # 3) keyword ALL-match
    if keywords:
        kw = [norm(k) for k in keywords if k]
        for c in cols:
            nc = norm(c)
            if all(k in nc for k in kw):
                return c
    # 4) contains heuristic
    for name in candidates or []:
        token = norm(name)
        for c in cols:
            if token and token in norm(c):
                return c
    return default

def _rb_norm_email(s):
    if _rb_pd.isna(s): return _rb_pd.NA
    return str(s).strip().lower()

def _render_marketing_referral_box(df_f, create_col, pay_col, source_col, country_col, agent_col):
    _rb_st.subheader("Referral Box")
    _rb_st.caption("Analyze **Self-Generated** referrals and identify how many were referred by a **paid learner**.")

    if df_f is None or df_f.empty:
        _rb_st.info("No data found in the main dataframe."); return

    # Resolve columns (include 'Referred Intent Source' per your definition)
    _create = create_col if (create_col in df_f.columns) else _rb_find_col(df_f, [
        "Deal Create Date","Create Date","Created Date","CreateDate","Created On"
    ])
    _pay    = pay_col    if (pay_col    in df_f.columns) else _rb_find_col(df_f, [
        "Payment Received Date","Payment Date","Enroll Date","Enrolment Date","Enrollment Date"
    ])
    _source = _rb_find_col(df_f, ["Referred Intent Source","Referral Intent Source","Referral Intent","Referral Source","Intent Source", source_col])
    _parent_email = _rb_find_col(df_f, ["Parent Email ID","Parent Email","Email","Parent_Email_ID","Parent Email Id"])
    _ref_by_email = _rb_find_col(df_f, ["Referred Parent Referred By Email","Parent Referred By Email","Referrer Email","Referred By Email","Parent Email Referred By","Parent Referred By","Referral Referred By Email","Referrer Parent Email","Referred Parent By Email","Ref By Email","Ref By Parent Email"], keywords=["referred","by","email"])
    _country = country_col if (country_col in df_f.columns) else _rb_find_col(df_f, ["Country Name","Country"])
    _agent   = agent_col   if (agent_col   in df_f.columns) else _rb_find_col(df_f, ["Academic Counselor","Agent","Caller","Counsellor","Counselor"])

    # Missing-check except referrer email (we allow UI fallback for that)
    missing = [n for n,v in {
        "Deal Create Date": _create,
        "Payment Received Date": _pay,
        "Referred/Referral Intent Source": _source,
        "Parent Email ID": _parent_email,
    }.items() if v is None]
    if missing:
        _rb_st.error("Missing required columns: " + ", ".join(missing)); return

    df = df_f.copy()
    # Parse dates
    for c in [_create, _pay]:
        if c and c in df.columns:
            df[c] = _rb_pd.to_datetime(df[c], errors="coerce")
    # Normalize emails
    for c in [_parent_email, _ref_by_email]:
        if c and c in df.columns:
            df[c] = df[c].map(_rb_norm_email)

    # If we couldn't detect referrer email, ask the user
    if _ref_by_email is None or _ref_by_email not in df.columns:
        _rb_st.warning("Referrer Email column was not detected automatically. Please select it below.")
        email_like_cols = [c for c in df.columns if ("mail" in c.lower()) or ("referred" in c.lower())]
        _ref_by_email = _rb_st.selectbox("Select the 'Referrer Email' column", options=email_like_cols or df.columns.tolist(), key="rb_ref_by_picker")
        if not _ref_by_email:
            _rb_st.stop()

    # UI: scope + presets + filters
    c1, c2, c3 = _rb_st.columns([1,1,1])
    with c1:
        scope = _rb_st.radio("Scope", ["Self-Generated","All Referrals"], index=0, key="rb_scope")
    with c2:
        preset = _rb_st.selectbox("Date preset", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, key="rb_preset")
    with c3:
        today = _rb_pd.Timestamp.now(tz="Asia/Kolkata").date()
        start_default, end_default = _rb_date_preset_bounds(preset, today=today)
        start_date = _rb_st.date_input("Start date", value=start_default, key="rb_start")
        end_date = _rb_st.date_input("End date", value=end_default, key="rb_end")
    if start_date > end_date: start_date, end_date = end_date, start_date

    row2 = _rb_st.columns([1,1,1])
    with row2[0]:
        paid_rule = _rb_st.selectbox("Referrer timing rule", ["Paid at any time","Paid before referral create"], index=0, key="rb_paid_rule")
    with row2[1]:
        sel_agents = _rb_st.multiselect("Agent(s)", sorted(df[_agent].dropna().unique().tolist()) if _agent else [], key="rb_agents")
    with row2[2]:
        sel_countries = _rb_st.multiselect("Country", sorted(df[_country].dropna().unique().tolist()) if _country else [], key="rb_countries")

    # Scope filter (STRICT per your definition)
    mask = _rb_pd.Series(True, index=df.index)
    if scope == "Self-Generated":
        _sg_norm = df[_source].astype(str).str.strip().str.lower().str.replace("-", " ", regex=False)
        mask &= _sg_norm.eq("self generated")
    # Date window
    if _create:
        mask &= df[_create].dt.date.between(start_date, end_date)
    # Agent & Country
    if _agent and sel_agents:
        mask &= df[_agent].isin(sel_agents)
    if _country and sel_countries:
        mask &= df[_country].isin(sel_countries)

    base = df[mask].copy()

    # Build referrer index from full dataset
    ref = df[[_parent_email, _pay]].copy().rename(columns={_parent_email:"_rb_parent_email", _pay:"_rb_pay"})
    ref["_rb_paid"] = ref["_rb_pay"].notna()
    grp = ref.groupby("_rb_parent_email", dropna=True).agg(
        _rb_first_enroll=("_rb_pay","min"),
        _rb_paid_ever=("_rb_paid","max"),
    ).reset_index()

    base["_rb_referrer"] = base[_ref_by_email]
    base = base.merge(grp, left_on="_rb_referrer", right_on="_rb_parent_email", how="left")

    if paid_rule == "Paid before referral create":
        base["_rb_paid_ref"] = (base["_rb_paid_ever"] == True) & (base["_rb_first_enroll"].notna()) & (base["_rb_first_enroll"] <= base[_create])
    else:
        base["_rb_paid_ref"] = (base["_rb_paid_ever"] == True)

    total = len(base)
    paid_count = int(base["_rb_paid_ref"].sum())
    not_paid = int(total - paid_count)
    pct_paid = (paid_count/total*100.0) if total else 0.0
    uniq_paid_referrers = base.loc[base["_rb_paid_ref"], "_rb_referrer"].nunique()

    k1,k2,k3,k4 = _rb_st.columns(4)
    k1.metric("Total referrals (scope)", f"{total:,}")
    k2.metric("Referred by paid learners", f"{paid_count:,}")
    k3.metric("Not paid referrers", f"{not_paid:,}")
    k4.metric("% by paid referrers", f"{pct_paid:.1f}%")

    _rb_st.markdown("### Summary by Referrer Status")
    sum_df = _rb_pd.DataFrame({"Status":["Paid Referrer","Not Paid"], "Referrals":[paid_count, not_paid]})
    sum_df["% Share"] = (sum_df["Referrals"] / sum_df["Referrals"].sum() * 100.0).round(1) if sum_df["Referrals"].sum()>0 else 0.0
    _rb_st.dataframe(sum_df, use_container_width=True)
    _rb_st.download_button("Download Summary (CSV)", sum_df.to_csv(index=False).encode("utf-8"), "referral_box_summary.csv", "text/csv", key="rb_dl_sum")

    _rb_st.markdown("### Referrer Details")
    det = base.groupby(["_rb_referrer","_rb_paid_ref"], dropna=False).agg(
        Referrals=( _parent_email, "count"),
        First_Enrollment_Date=("_rb_first_enroll","min"),
        Last_Referral_Date=(_create,"max")
    ).reset_index().rename(columns={"_rb_referrer":"Referrer Email","_rb_paid_ref":"Referrer Paid?"})
    _rb_st.dataframe(det, use_container_width=True)
    _rb_st.download_button("Download Referrer Details (CSV)", det.to_csv(index=False).encode("utf-8"), "referrer_details.csv", "text/csv", key="rb_dl_det")

    _rb_st.markdown("### Referral Deals (row-level)")
    view_cols = []
    for c in ["Record ID","RecordID","ID", _create, _parent_email, _ref_by_email, _country, _agent, _source]:
        if c and c in base.columns and c not in view_cols:
            view_cols.append(c)
    base["_rb_referrer_paid?"] = base["_rb_paid_ref"].map({True:"Yes", False:"No", _rb_pd.NA:_rb_pd.NA})
    base["_rb_days_between"] = _rb_pd.to_timedelta(base[_create] - base["_rb_first_enroll"]).dt.days
    derived = ["_rb_referrer_paid?","_rb_first_enroll","_rb_days_between"]
    rows = base[view_cols + derived].copy().rename(columns={
        _create:"Deal Create Date",
        _parent_email:"Parent Email ID",
        _ref_by_email:"Referrer Email",
        _country:"Country",
        _agent:"Academic Counselor",
        _source:"Referred Intent Source",
        "_rb_first_enroll":"Referrer First Enrollment Date",
        "_rb_referrer_paid?":"Referrer Paid?",
        "_rb_days_between":"Days Between (Enroll → Referral)"
    })
    _rb_st.dataframe(rows, use_container_width=True)
    _rb_st.download_button("Download Referral Deals (CSV)", rows.to_csv(index=False).encode("utf-8"), "referral_deals.csv", "text/csv", key="rb_dl_rows")

# ============================
# END: Marketing -> Referral Box (appended by patch)
# ============================



# --- Router: Marketing → Talk Time ---
try:
    _master_for_tt = master if 'master' in globals() else st.session_state.get('nav_master', '')
    _view_for_tt = view if 'view' in globals() else st.session_state.get('nav_sub', '')
    if _master_for_tt == "Marketing" and _view_for_tt == "Talk Time":
        try:
            _render_marketing_talk_time(df)
        except TypeError:
            _render_marketing_talk_time(df_f if 'df_f' in globals() else None)
except Exception as _e_tt:
    import streamlit as st
    st.error(f"Talk Time failed: {type(_e_tt).__name__}: {_e_tt}")
# --- /Router: Marketing → Talk Time ---




# =============  CLOSED LOST ANALYSIS — Funnel & Movement  =============
import pandas as _pd
import numpy as _np
import altair as _alt
from datetime import date as _date, timedelta as _timedelta

def _resolve_col(df, preferred, candidates):
    if isinstance(preferred, str) and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        k = c.lower().strip()
        if k in low:
            return low[k]
    return None

def _to_dt(series):
    try:
        s = _pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    except Exception:
        s = _pd.to_datetime(series, errors="coerce")
    mask = s.isna()
    if mask.any():
        raw = series.astype(str).str.strip()
        m2 = raw.str.fullmatch(r"\d{8}", na=False)
        s2 = _pd.to_datetime(raw.where(m2), format="%d%m%Y", errors="coerce")
        s = s.fillna(s2)
        mask = s.isna()
        if mask.any():
            s3 = _pd.to_datetime(raw.where(mask).str.slice(0, 10), errors="coerce", dayfirst=True)
            s = s.fillna(s3)
    return s

def _normalize_text(s):
    return s.fillna("Unknown").astype(str).str.strip()

def _classify_interval(row, cols):
    cl = row[cols["closed_lost"]]
    if _pd.isna(cl):
        return ("No Closed Lost Date", _np.nan)

    c  = row.get(cols.get("create"))
    ts = row.get(cols.get("trial_s"))
    tr = row.get(cols.get("trial_r"))
    cd = row.get(cols.get("cal_done"))

    steps = []
    if not _pd.isna(c):  steps.append(("Create", c))
    if not _pd.isna(ts): steps.append(("Trial Scheduled", ts))
    if not _pd.isna(tr): steps.append(("Trial Rescheduled", tr))
    if not _pd.isna(cd): steps.append(("Calibration Done", cd))

    if not steps:
        return ("Before Create / No Milestones", _np.nan)

    first_name, first_date = steps[0]
    if cl < first_date:
        return ("Before Create / No Milestones", (cl - first_date).days)

    for i in range(len(steps) - 1):
        name_a, d_a = steps[i]
        name_b, d_b = steps[i + 1]
        if (cl >= d_a) and (cl < d_b):
            return (f"{name_a} → {name_b}", (cl - d_a).days)

    last_name, last_date = steps[-1]
    if cl >= last_date:
        return ("After Calibration Done" if last_name == "Calibration Done" else f"After {last_name}", (cl - last_date).days)

    return ("Unclassified", _np.nan)

def _render_funnel_closed_lost_analysis(df_f, counsellor_col_hint=None, country_col_hint=None, source_col_hint=None,
                                        create_col_hint=None, trial_s_col_hint=None, trial_r_col_hint=None,
                                        cal_done_col_hint=None, closed_lost_col_hint=None):
    import streamlit as st

    st.subheader("Funnel & Movement — Closed Lost Analysis")

    _counsellor = _resolve_col(df_f, counsellor_col_hint, ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor","Deal Owner"])
    _country    = _resolve_col(df_f, country_col_hint,    ["Country","Country Name"])
    _source     = _resolve_col(df_f, source_col_hint,     ["JetLearn Deal Source","Deal Source","Source","Original source"])
    _create     = _resolve_col(df_f, create_col_hint,     ["Create Date","Created Date","Deal Create Date","CreateDate","Created On","Creation Date"])
    _trial_s    = _resolve_col(df_f, trial_s_col_hint,    ["Trial Scheduled Date","Trial Schedule Date","Trial Booking Date","First Calibration Scheduled Date"])
    _trial_r    = _resolve_col(df_f, trial_r_col_hint,    ["Trial Rescheduled Date","Trial Re-scheduled Date","Calibration Rescheduled Date"])
    _cal_done   = _resolve_col(df_f, cal_done_col_hint,   ["Calibration Done Date","Calibration Completed Date","Trial Done Date"])
    _cl_date    = _resolve_col(df_f, closed_lost_col_hint,["[Deal Stage] - Closed Lost Trigger Date","Closed Lost Trigger Date","Closed Lost Date","Closed-Lost Trigger Date"])

    if _cl_date is None:
        st.warning("‘Closed Lost Trigger Date’ column not found.", icon="⚠️"); return
    if _create is None:
        st.warning("‘Create Date’ column not found.", icon="⚠️"); return

    d = df_f.copy()
    d["_CL"]  = _to_dt(d[_cl_date])
    d["_C"]   = _to_dt(d[_create])
    d["_TS"]  = _to_dt(d[_trial_s]) if _trial_s else _pd.NaT
    d["_TR"]  = _to_dt(d[_trial_r]) if _trial_r else _pd.NaT
    d["_CD"]  = _to_dt(d[_cal_done]) if _cal_done else _pd.NaT

    d["_AC"]   = _normalize_text(d[_counsellor]) if _counsellor else _pd.Series(["Unknown"]*len(d))
    d["_CNT"]  = _normalize_text(d[_country]) if _country else _pd.Series(["Unknown"]*len(d))
    d["_SRC"]  = _normalize_text(d[_source]) if _source else _pd.Series(["Unknown"]*len(d))

    c1,c2,c3,c4 = st.columns([1,1,1,2])
    with c1:
        mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="cl_mode")
    with c2:
        scope = st.radio("Date scope", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, horizontal=True, key="cl_scope")
    with c3:
        chart_type = st.radio("Chart", ["Stacked Bar","Line"], index=0, horizontal=True, key="cl_chart")
    with c4:
        dim_opts = []
        if _counsellor: dim_opts.append("Academic Counsellor")
        if _country:    dim_opts.append("Country")
        if _source:     dim_opts.append("JetLearn Deal Source")
        sel_dims = st.multiselect("Dimensions (choose 1–2 for best visuals)", options=dim_opts, default=(dim_opts[:1] if dim_opts else []), key="cl_dims")

    today = _date.today()
    def _month_bounds(d0: _date):
        from calendar import monthrange
        start = _date(d0.year, d0.month, 1)
        end = _date(d0.year, d0.month, monthrange(d0.year, d0.month)[1])
        return start, end

    if scope == "Today":
        start_d, end_d = today, today
    elif scope == "Yesterday":
        start_d = today - _timedelta(days=1); end_d = start_d
    elif scope == "This Month":
        start_d, end_d = _month_bounds(today)
    elif scope == "Last Month":
        first_this = _date(today.year, today.month, 1)
        last_prev  = first_this - _timedelta(days=1)
        start_d, end_d = _month_bounds(last_prev)
    else:
        cc1, cc2 = st.columns(2)
        with cc1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="cl_start")
        with cc2: end_d   = st.date_input("End", value=today, key="cl_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    cl_in = d["_CL"].dt.date.between(start_d, end_d)
    if mode == "MTD":
        c_in  = d["_C"].dt.date.between(start_d, end_d)
        in_scope = cl_in & c_in
    else:
        in_scope = cl_in

    dd = d.loc[in_scope].copy()
    st.caption(f"Window: **{start_d} → {end_d}** • Mode: **{mode}**  • Rows in scope: **{len(dd)}**")

    if dd.empty:
        st.info("No Closed Lost records in the selected window."); return

    st.markdown("### A) Closed Lost — Counts by Dimension")
    def _map_dim(name):
        if name == "Academic Counsellor": return "_AC"
        if name == "Country": return "_CNT"
        if name == "JetLearn Deal Source": return "_SRC"
        return None

    dim_cols = [_map_dim(x) for x in sel_dims if _map_dim(x)]
    if not dim_cols:
        dd["_All"] = "All"; dim_cols = ["_All"]

    agg_a = dd.groupby(dim_cols, dropna=False).size().reset_index(name="Closed Lost Count")

    if chart_type == "Stacked Bar":
        if len(dim_cols) == 1:
            ch_a = _alt.Chart(agg_a).mark_bar().encode(
                x=_alt.X(f"{dim_cols[0]}:N", title=sel_dims[0] if sel_dims else "All"),
                y=_alt.Y("Closed Lost Count:Q"),
                tooltip=[_alt.Tooltip(f"{dim_cols[0]}:N", title=sel_dims[0] if sel_dims else "All"),
                         _alt.Tooltip("Closed Lost Count:Q")]
            ).properties(height=340)
        else:
            ch_a = _alt.Chart(agg_a).mark_bar().encode(
                x=_alt.X(f"{dim_cols[0]}:N", title=sel_dims[0]),
                y=_alt.Y("Closed Lost Count:Q"),
                color=_alt.Color(f"{dim_cols[1]}:N", title=sel_dims[1]),
                tooltip=[_alt.Tooltip(f"{dim_cols[0]}:N", title=sel_dims[0]),
                         _alt.Tooltip(f"{dim_cols[1]}:N", title=sel_dims[1]),
                         _alt.Tooltip("Closed Lost Count:Q")]
            ).properties(height=340)
        st.altair_chart(ch_a, use_container_width=True)
    else:
        dd["_d"] = dd["_CL"].dt.date
        ts = dd.groupby(["_d"] + dim_cols, dropna=False).size().reset_index(name="Closed Lost Count")
        color_enc = _alt.Color(f"{dim_cols[0]}:N", title=(sel_dims[0] if sel_dims else "All")) if dim_cols else _alt.value("steelblue")
        ch_a = _alt.Chart(ts).mark_line(point=True).encode(
            x=_alt.X("_d:T", title=None),
            y=_alt.Y("Closed Lost Count:Q"),
            color=color_enc,
            tooltip=[_alt.Tooltip("_d:T", title="Date"),
                     _alt.Tooltip("Closed Lost Count:Q")] + (
                        [_alt.Tooltip(f"{dim_cols[0]}:N", title=sel_dims[0])] if len(dim_cols) >= 1 else []
                     )
        ).properties(height=340)
        st.altair_chart(ch_a, use_container_width=True)

    st.dataframe(agg_a, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Closed Lost by Dimension",
                       data=agg_a.to_csv(index=False).encode("utf-8"),
                       file_name="closed_lost_by_dimension.csv",
                       mime="text/csv",
                       key="cl_dl_a")

    st.markdown("### B) Where in the Journey do Deals get Lost? (Milestone Intervals)")
    cols_map = {"closed_lost":"_CL","create":"_C","trial_s":"_TS","trial_r":"_TR","cal_done":"_CD"}
    dd["_interval"], dd["_days_to_loss"] = zip(*dd.apply(lambda r: _classify_interval(r, cols_map), axis=1))

    by_cols = ["_interval"]; label_interval = "Interval Bucket"
    if dim_cols: by_cols.append(dim_cols[0])
    tmp = dd.groupby(by_cols, dropna=False).size().reset_index(name="Count")
    agg_b = tmp.merge(
        dd.groupby(by_cols, dropna=False)["_days_to_loss"].mean().reset_index(name="Avg_Days_to_Loss"),
        on=by_cols, how="left"
    )
    agg_b["Avg_Days_to_Loss"] = agg_b["Avg_Days_to_Loss"].round(1)

    color_enc = _alt.Color(f"{dim_cols[0]}:N", title=(sel_dims[0])) if dim_cols else _alt.value(None)
    ch_b = _alt.Chart(agg_b).mark_circle().encode(
        x=_alt.X("_interval:N", title=label_interval, sort=[
            "Before Create / No Milestones",
            "Create → Trial Scheduled",
            "Trial Scheduled → Trial Rescheduled",
            "Trial Rescheduled → Calibration Done",
            "After Calibration Done"
        ]),
        y=_alt.Y("Count:Q"),
        size=_alt.Size("Avg_Days_to_Loss:Q", legend=_alt.Legend(title="Avg days to loss")),
        color=color_enc,
        tooltip=[_alt.Tooltip("_interval:N", title=label_interval),
                 _alt.Tooltip("Count:Q"),
                 _alt.Tooltip("Avg_Days_to_Loss:Q")] + (
                    [_alt.Tooltip(f"{dim_cols[0]}:N", title=sel_dims[0])] if dim_cols else []
                 )
    ).properties(height=360)
    st.altair_chart(ch_b, use_container_width=True)

    pretty_cols = {}
    if dim_cols: pretty_cols[dim_cols[0]] = sel_dims[0]
    tbl_b = agg_b.rename(columns={"_interval": label_interval, **pretty_cols})
    st.dataframe(tbl_b, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Journey Interval Analysis",
                       data=tbl_b.to_csv(index=False).encode("utf-8"),
                       file_name="closed_lost_interval_analysis.csv",
                       mime="text/csv",
                       key="cl_dl_b")

try:
    if "MASTER_SECTIONS" in globals() and isinstance(MASTER_SECTIONS, dict):
        fm = MASTER_SECTIONS.get("Funnel & Movement", [])
        if "Closed Lost Analysis" not in fm:
            MASTER_SECTIONS["Funnel & Movement"] = fm + ["Closed Lost Analysis"]
except Exception:
    pass

try:
    if view == "Closed Lost Analysis":
        _render_funnel_closed_lost_analysis(df,
            counsellor_col_hint=locals().get("counsellor_col", None),
            country_col_hint=locals().get("country_col", None),
            source_col_hint=locals().get("source_col", None),
            create_col_hint=locals().get("create_col", None),
            trial_s_col_hint=locals().get("first_cal_sched_col", None) or locals().get("trial_s_col", None),
            trial_r_col_hint=locals().get("cal_resched_col", None) or locals().get("trial_r_col", None),
            cal_done_col_hint=locals().get("cal_done_col", None),
            closed_lost_col_hint=None)
except Exception as _e:
    import streamlit as st
    st.error(f"Closed Lost Analysis failed: {type(_e).__name__}: {_e}")
# =============  /CLOSED LOST ANALYSIS  =============


# =============  BOOKING ANALYSIS — Funnel & Movement  =============
import pandas as _pd
import numpy as _np
import altair as _alt
from datetime import date as _date, timedelta as _timedelta

def _bk_resolve_col(df, preferred, candidates):
    if isinstance(preferred, str) and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        k = c.lower().strip()
        if k in low:
            return low[k]
    return None

def _bk_to_dt(series):
    try:
        s = _pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    except Exception:
        s = _pd.to_datetime(series, errors="coerce")
    mask = s.isna()
    if mask.any():
        raw = series.astype(str).str.strip()
        m2 = raw.str.fullmatch(r"\d{8}", na=False)
        s2 = _pd.to_datetime(raw.where(m2), format="%d%m%Y", errors="coerce")
        s = s.fillna(s2)
        mask = s.isna()
        if mask.any():
            s3 = _pd.to_datetime(raw.where(mask).str.slice(0, 10), errors="coerce", dayfirst=True)
            s = s.fillna(s3)
    return s

def _render_funnel_booking_analysis(df_f, trigger_book_col_hint=None, cal_slot_col_hint=None, first_cal_sched_col_hint=None,
                                    counsellor_col_hint=None, country_col_hint=None, source_col_hint=None):
    import streamlit as st

    st.subheader("Funnel & Movement — Booking Analysis")

    _trigger  = _bk_resolve_col(df_f, trigger_book_col_hint, ["[Trigger] - Calibration Booking Date","Trigger - Calibration Booking Date","Calibration Booking Date"])
    _slot     = _bk_resolve_col(df_f, cal_slot_col_hint, ["Calibration Slot (Deal)","Calibration Slot","Booking Slot (Deal)"])
    _first    = _bk_resolve_col(df_f, first_cal_sched_col_hint, ["First Calibration Scheduled Date","Trial Scheduled Date","Trial Schedule Date"])
    _cns      = _bk_resolve_col(df_f, counsellor_col_hint, ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor","Deal Owner"])
    _cty      = _bk_resolve_col(df_f, country_col_hint, ["Country","Country Name"])
    _src      = _bk_resolve_col(df_f, source_col_hint, ["JetLearn Deal Source","Deal Source","Source","Original source"])
    _create   = _bk_resolve_col(df_f, None, ["Create Date","Created Date","Deal Create Date","CreateDate","Created On","Creation Date"])

    if _trigger is None:
        st.warning("‘[Trigger] - Calibration Booking Date’ column not found.", icon="⚠️"); return
    if _first is None and _slot is None:
        st.warning("Neither ‘Calibration Slot (Deal)’ nor ‘First Calibration Scheduled Date’ found.", icon="⚠️"); return

    d = df_f.copy()
    d["_TRIG"] = _bk_to_dt(d[_trigger])
    d["_SLOT"] = d[_slot].astype(str).str.strip() if _slot else _pd.Series([""]*len(d))
    d["_FIRST"]= _bk_to_dt(d[_first]) if _first else _pd.NaT
    d["_C"]    = _bk_to_dt(d[_create]) if _create else _pd.NaT

    pre_mask  = d["_SLOT"].notna() & (d["_SLOT"].str.len() > 0) & (d["_SLOT"].str.lower() != "nan")
    self_mask = (~pre_mask) & d["_FIRST"].notna()
    d["_BKTYPE"] = _np.select([pre_mask, self_mask], ["Pre-book","Self book"], default="Unknown")

    d["_AC"]  = d[_cns].fillna("Unknown").astype(str).str.strip() if _cns else _pd.Series(["Unknown"]*len(d))
    d["_CNT"] = d[_cty].fillna("Unknown").astype(str).str.strip() if _cty else _pd.Series(["Unknown"]*len(d))
    d["_SRC"] = d[_src].fillna("Unknown").astype(str).str.strip() if _src else _pd.Series(["Unknown"]*len(d))

    # Derived slice flags
    _resch_col  = _bk_resolve_col(df_f, None, ["Trial Rescheduled Date","Trial Re-scheduled Date","Calibration Rescheduled Date"])
    _caldone_col= _bk_resolve_col(df_f, None, ["Calibration Done Date","Calibration Completed Date","Trial Done Date"])
    _enrol_col  = _bk_resolve_col(df_f, None, ["Payment Received Date","Enrollment Date","Enrolment Date","Payment Date"])

    d["_HAS_FIRST"]   = _pd.Series(_pd.notna(d["_FIRST"]).map({True:"Yes", False:"No"}))
    d["_HAS_RESCH"]   = _pd.Series(_pd.notna(_bk_to_dt(d[_resch_col])) .map({True:"Yes", False:"No"})) if _resch_col else _pd.Series(["No"]*len(d))
    d["_HAS_CALDONE"] = _pd.Series(_pd.notna(_bk_to_dt(d[_caldone_col])).map({True:"Yes", False:"No"})) if _caldone_col else _pd.Series(["No"]*len(d))
    d["_HAS_ENRL"]    = _pd.Series(_pd.notna(_bk_to_dt(d[_enrol_col]))  .map({True:"Yes", False:"No"})) if _enrol_col else _pd.Series(["No"]*len(d))

    # Controls
    c0,c1,c2,c3 = st.columns([1.0,1.0,1.0,1.6])
    with c0:
        mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="bk_mode")
    with c1:
        scope = st.radio("Date scope", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, horizontal=True, key="bk_scope")
    with c2:
        gran = st.radio("Granularity", ["Daily","Monthly"], index=0, horizontal=True, key="bk_gran")
    with c3:
        dims = st.multiselect("Slice by", options=["Academic Counsellor","Country","JetLearn Deal Source","Booking Type","First Trial","Trial Reschedule","Calibration Done","Enrolment"],
                              default=["Booking Type"], key="bk_dims")

    # Date window
    today = _date.today()
    def _month_bounds(d0: _date):
        from calendar import monthrange
        start = _date(d0.year, d0.month, 1)
        end = _date(d0.year, d0.month, monthrange(d0.year, d0.month)[1])
        return start, end

    if scope == "Today":
        start_d, end_d = today, today
    elif scope == "Yesterday":
        start_d = today - _timedelta(days=1); end_d = start_d
    elif scope == "This Month":
        start_d, end_d = _month_bounds(today)
    elif scope == "Last Month":
        first_this = _date(today.year, today.month, 1)
        last_prev  = first_this - _timedelta(days=1)
        start_d, end_d = _month_bounds(last_prev)
    else:
        cst, cen = st.columns(2)
        with cst: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="bk_start")
        with cen: end_d   = st.date_input("End", value=today, key="bk_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Scope logic: Trigger in window, and if MTD then also Create in window
    cl_in = d["_TRIG"].dt.date.between(start_d, end_d)
    if mode == "MTD" and _create is not None:
        c_in = d["_C"].dt.date.between(start_d, end_d)
        in_win = cl_in & c_in
    else:
        in_win = cl_in
    dfw = d.loc[in_win].copy()

    # Filters
    fc1,fc2,fc3,fc4 = st.columns([1.2,1.2,1.2,1.2])
    with fc1:
        ac_opts = ["All"] + sorted(dfw["_AC"].unique().tolist())
        pick_ac = st.multiselect("Academic Counsellor", options=ac_opts, default=["All"], key="bk_ac")
    with fc2:
        ctry_opts = ["All"] + sorted(dfw["_CNT"].unique().tolist())
        pick_cty = st.multiselect("Country", options=ctry_opts, default=["All"], key="bk_cty")
    with fc3:
        src_opts = ["All"] + sorted(dfw["_SRC"].unique().tolist())
        pick_src = st.multiselect("JetLearn Deal Source", options=src_opts, default=["All"], key="bk_src")
    with fc4:
        bkt_opts = ["All"] + ["Pre-book","Self book","Unknown"]
        pick_bkt = st.multiselect("Booking Type", options=bkt_opts, default=["Pre-book","Self book"], key="bk_bkt")

    def _resolve(vals, all_vals):
        return all_vals if ("All" in vals or not vals) else vals

    ac_sel  = _resolve(pick_ac, sorted(dfw["_AC"].unique().tolist()))
    cty_sel = _resolve(pick_cty, sorted(dfw["_CNT"].unique().tolist()))
    src_sel = _resolve(pick_src, sorted(dfw["_SRC"].unique().tolist()))
    bkt_sel = _resolve(pick_bkt, ["Pre-book","Self book","Unknown"])

    mask = dfw["_AC"].isin(ac_sel) & dfw["_CNT"].isin(cty_sel) & dfw["_SRC"].isin(src_sel) & dfw["_BKTYPE"].isin(bkt_sel)
    dfw = dfw.loc[mask].copy()

    st.caption(f"Window: **{start_d} → {end_d}** • Mode: **{mode}**  • Rows: **{len(dfw)}**")
    if dfw.empty:
        st.info("No records for selected filters/date range."); return

    # Trend build
    dfw["_day"] = dfw["_TRIG"].dt.date
    dfw["_mon"] = _pd.to_datetime(dfw["_TRIG"].dt.to_period("M").astype(str))

    def _map_dim(x):
        return {"Academic Counsellor":"_AC", "Country":"_CNT", "JetLearn Deal Source":"_SRC", "Booking Type":"_BKTYPE",
                "First Trial":"_HAS_FIRST", "Trial Reschedule":"_HAS_RESCH", "Calibration Done":"_HAS_CALDONE", "Enrolment":"_HAS_ENRL"}.get(x)

    dim_cols = [_map_dim(x) for x in dims if _map_dim(x)]
    if not dim_cols:
        dfw["_All"] = "All"; dim_cols = ["_All"]

    if gran == "Daily":
        grp_cols = ["_day"] + dim_cols
        series = dfw.groupby(grp_cols, dropna=False).size().rename("Count").reset_index()
        x_enc = _alt.X("_day:T", title=None)
    else:
        grp_cols = ["_mon"] + dim_cols
        series = dfw.groupby(grp_cols, dropna=False).size().rename("Count").reset_index()
        x_enc = _alt.X("_mon:T", title=None)

    color_enc = _alt.Color(f"{dim_cols[0]}:N", title=dims[0] if dims else "All")
    ch = (_alt.Chart(series).mark_bar().encode(x=x_enc, y=_alt.Y("Count:Q"), color=color_enc,
          tooltip=[_alt.Tooltip("Count:Q")]).properties(height=320))
    st.altair_chart(ch, use_container_width=True)

    pretty = series.rename(columns={"_day":"Date","_mon":"Month","_AC":"Academic Counsellor","_CNT":"Country","_SRC":"JetLearn Deal Source","_BKTYPE":"Booking Type"})
    st.dataframe(pretty, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Booking Analysis Trend", data=pretty.to_csv(index=False).encode("utf-8"),
                       file_name="booking_analysis_trend.csv", mime="text/csv", key="bk_dl_tbl")

    # Comparison vs Trial Scheduled
    st.markdown("### Comparison: Booking vs Trial Scheduled")
    do_compare = st.checkbox("Show comparison vs Trial Scheduled", value=True, key="bk_compare_toggle")
    if do_compare:
        ts = d.copy()
        ts = ts[ts["_FIRST"].notna()].copy()
        ts_in = ts["_FIRST"].dt.date.between(start_d, end_d)
        if mode == "MTD" and _create is not None:
            cts_in = ts["_C"].dt.date.between(start_d, end_d)
            ts_in = ts_in & cts_in
        ts = ts.loc[ts_in].copy()
        ts["_AC"] = ts["_AC"].fillna("Unknown"); ts["_CNT"] = ts["_CNT"].fillna("Unknown"); ts["_SRC"] = ts["_SRC"].fillna("Unknown")
        ts_mask = ts["_AC"].isin(ac_sel) & ts["_CNT"].isin(cty_sel) & ts["_SRC"].isin(src_sel)
        ts = ts.loc[ts_mask].copy()

        if gran == "Daily":
            ts["_day"] = ts["_FIRST"].dt.date
            ts_series = ts.groupby(["_day"], dropna=False).size().rename("Count").reset_index()
            ts_series["_x"] = ts_series["_day"]
            bk_series = series.copy()
            if "_day" in bk_series.columns: bk_series["_x"] = bk_series["_day"]
            else:
                tmp = dfw.groupby(["_day"], dropna=False).size().rename("Count").reset_index()
                bk_series = tmp.assign(_x=tmp["_day"])
        else:
            ts["_mon"] = _pd.to_datetime(ts["_FIRST"].dt.to_period("M").astype(str))
            ts_series = ts.groupby(["_mon"], dropna=False).size().rename("Count").reset_index()
            ts_series["_x"] = ts_series["_mon"]
            bk_series = series.copy()
            if "_mon" in bk_series.columns: bk_series["_x"] = bk_series["_mon"]
            else:
                tmp = dfw.groupby(["_mon"], dropna=False).size().rename("Count").reset_index()
                bk_series = tmp.assign(_x=tmp["_mon"])

        ts_series["Series"] = "Trial Scheduled"
        bk_series_slim = bk_series[["_x","Count"]].copy(); bk_series_slim["Series"] = "Booking Trigger"
        comp = _pd.concat([bk_series_slim, ts_series[["_x","Count","Series"]]], ignore_index=True).sort_values(["_x","Series"]).reset_index(drop=True)

        ch_cmp = (_alt.Chart(comp).mark_line(point=True).encode(
                    x=_alt.X("_x:T", title=None), y=_alt.Y("Count:Q"),
                    color=_alt.Color("Series:N", title="Series"),
                    tooltip=[_alt.Tooltip("_x:T", title="Date/Month"), _alt.Tooltip("Series:N"), _alt.Tooltip("Count:Q")]
                 ).properties(height=320))
        st.altair_chart(ch_cmp, use_container_width=True)

        pretty_cmp = comp.rename(columns={"_x":"Date" if gran=="Daily" else "Month"})
        st.download_button("Download CSV — Booking vs Trial Scheduled", data=pretty_cmp.to_csv(index=False).encode("utf-8"),
                           file_name="booking_vs_trial_scheduled.csv", mime="text/csv", key="bk_dl_cmp")

# Ensure pill in menu
try:
    if "MASTER_SECTIONS" in globals() and isinstance(MASTER_SECTIONS, dict):
        fm = MASTER_SECTIONS.get("Funnel & Movement", [])
        for pill in ["Closed Lost Analysis","Booking Analysis"]:
            if pill not in fm:
                fm = fm + [pill]
        MASTER_SECTIONS["Funnel & Movement"] = fm
except Exception:
    pass

try:
    if view == "Booking Analysis":
        _render_funnel_booking_analysis(df,
            trigger_book_col_hint=None,
            cal_slot_col_hint=None,
            first_cal_sched_col_hint=None,
            counsellor_col_hint=locals().get("counsellor_col", None),
            country_col_hint=locals().get("country_col", None),
            source_col_hint=locals().get("source_col", None))
except Exception as _e:
    import streamlit as st
    st.error(f"Booking Analysis failed: {type(_e).__name__}: {_e}")
# =============  /BOOKING ANALYSIS  =============


# =============  TRIAL TREND — Funnel & Movement  =============
import pandas as _pd
import numpy as _np
import altair as _alt
from datetime import date as _date, timedelta as _timedelta

def _tt_resolve_col(df, preferred, candidates):
    if isinstance(preferred, str) and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        k = c.lower().strip()
        if k in low:
            return low[k]
    return None

def _tt_to_dt(series):
    try:
        s = _pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    except Exception:
        s = _pd.to_datetime(series, errors="coerce")
    mask = s.isna()
    if mask.any():
        raw = series.astype(str).str.strip()
        m2 = raw.str.fullmatch(r"\d{8}", na=False)
        s2 = _pd.to_datetime(raw.where(m2), format="%d%m%Y", errors="coerce")
        s = s.fillna(s2)
        mask = s.isna()
        if mask.any():
            s3 = _pd.to_datetime(raw.where(mask).str.slice(0, 10), errors="coerce", dayfirst=True)
            s = s.fillna(s3)
    return s

def _render_funnel_trial_trend(
    df_f,
    first_trial_col_hint: str | None = None,
    resched_col_hint: str | None = None,
    trial_done_col_hint: str | None = None,
    enrol_col_hint: str | None = None,
    create_col_hint: str | None = None,
    counsellor_col_hint: str | None = None,
    country_col_hint: str | None = None,
    source_col_hint: str | None = None,
):
    import streamlit as st

    st.subheader("Funnel & Movement — Trial Trend")

    _first   = _tt_resolve_col(df_f, first_trial_col_hint, ["First Calibration Scheduled Date","Trial Scheduled Date","Trial Schedule Date"])
    _resch   = _tt_resolve_col(df_f, resched_col_hint,      ["Trial Rescheduled Date","Trial Re-scheduled Date","Calibration Rescheduled Date"])
    _done    = _tt_resolve_col(df_f, trial_done_col_hint,   ["Calibration Done Date","Calibration Completed Date","Trial Done Date"])
    _enrol   = _tt_resolve_col(df_f, enrol_col_hint,        ["Payment Received Date","Enrollment Date","Enrolment Date","Payment Date"])
    _create  = _tt_resolve_col(df_f, create_col_hint,       ["Create Date","Created Date","Deal Create Date","CreateDate","Created On","Creation Date"])
    _cns     = _tt_resolve_col(df_f, counsellor_col_hint,   ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor","Deal Owner"])
    _cty     = _tt_resolve_col(df_f, country_col_hint,      ["Country","Country Name"])
    _src     = _tt_resolve_col(df_f, source_col_hint,       ["JetLearn Deal Source","Deal Source","Source","Original source"])

    if _first is None and _resch is None:
        st.warning("Neither ‘First Trial’ nor ‘Trial Rescheduled’ columns found.", icon="⚠️"); return

    d = df_f.copy()
    d["_FT"]   = _tt_to_dt(d[_first]) if _first else _pd.NaT
    d["_TR"]   = _tt_to_dt(d[_resch]) if _resch else _pd.NaT
    d["_TDONE"]= _tt_to_dt(d[_done]) if _done else _pd.NaT
    d["_ENR"]  = _tt_to_dt(d[_enrol]) if _enrol else _pd.NaT
    d["_C"]    = _tt_to_dt(d[_create]) if _create else _pd.NaT

    d["_AC"]  = d[_cns].fillna("Unknown").astype(str).str.strip() if _cns else _pd.Series(["Unknown"]*len(d))
    d["_CNT"] = d[_cty].fillna("Unknown").astype(str).str.strip() if _cty else _pd.Series(["Unknown"]*len(d))
    d["_SRC"] = d[_src].fillna("Unknown").astype(str).str.strip() if _src else _pd.Series(["Unknown"]*len(d))

    # Controls
    c0,c1,c2,c3 = st.columns([1.0,1.0,1.0,1.6])
    with c0:
        mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="tt_mode")
    with c1:
        scope = st.radio("Date scope", ["Today","Yesterday","This Month","Last Month","Custom"], index=2, horizontal=True, key="tt_scope")
    with c2:
        gran = st.radio("Granularity", ["Daily","Monthly"], index=0, horizontal=True, key="tt_gran")
    with c3:
        dims = st.multiselect("Slice by", options=["Academic Counsellor","Country","JetLearn Deal Source"],
                              default=["Academic Counsellor"], key="tt_dims")

    # Date window
    today = _date.today()
    def _month_bounds(d0: _date):
        from calendar import monthrange
        start = _date(d0.year, d0.month, 1)
        end = _date(d0.year, d0.month, monthrange(d0.year, d0.month)[1])
        return start, end

    if scope == "Today":
        start_d, end_d = today, today
    elif scope == "Yesterday":
        start_d = today - _timedelta(days=1); end_d = start_d
    elif scope == "This Month":
        start_d, end_d = _month_bounds(today)
    elif scope == "Last Month":
        first_this = _date(today.year, today.month, 1)
        last_prev  = first_this - _timedelta(days=1)
        start_d, end_d = _month_bounds(last_prev)
    else:
        cst, cen = st.columns(2)
        with cst: start_d = st.date_input("Start", value=_month_bounds(today)[0], key="tt_start")
        with cen: end_d   = st.date_input("End", value=today, key="tt_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    # Build event rows for each metric
    events = []  # list of (date, metric, AC, CNT, SRC, create_date)
    # For each row, produce trial events:
    for idx, row in d.iterrows():
        ac, cnt, src = row["_AC"], row["_CNT"], row["_SRC"]
        cdate = row["_C"] if _create else _pd.NaT

        ft = row["_FT"]; tr = row["_TR"]; td = row["_TDONE"]; en = row["_ENR"]
        # Trial union: emit up to two events, but if FT and TR fall on the same day, emit only one
        trial_dates = set()
        if _pd.notna(ft): trial_dates.add(_pd.Timestamp(ft).normalize())
        if _pd.notna(tr): trial_dates.add(_pd.Timestamp(tr).normalize())
        for dt in sorted(trial_dates):
            events.append((dt, "Trial", ac, cnt, src, cdate))

        # Trial Done
        if _pd.notna(td):
            events.append((_pd.Timestamp(td).normalize(), "Trial Done", ac, cnt, src, cdate))

        # Enrollment
        if _pd.notna(en):
            events.append((_pd.Timestamp(en).normalize(), "Enrollment", ac, cnt, src, cdate))

        # Lead (Create)
        if _pd.notna(cdate):
            events.append((_pd.Timestamp(cdate).normalize(), "Lead", ac, cnt, src, cdate))

    if not events:
        st.info("No trial/trial-done/enrollment/lead events found."); return

    ev = _pd.DataFrame(events, columns=["_when","Metric","_AC","_CNT","_SRC","_C"])

    # Apply window per mode
    in_win = ev["_when"].dt.date.between(start_d, end_d)
    if mode == "MTD" and _create is not None:
        c_in = _pd.to_datetime(ev["_C"]).dt.date.between(start_d, end_d)
        in_win = in_win & c_in
    ev = ev.loc[in_win].copy()

    if ev.empty:
        st.info("No events in selected window/filters."); return

    # Filters
    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        ac_opts = ["All"] + sorted(ev["_AC"].unique().tolist())
        pick_ac = st.multiselect("Academic Counsellor", options=ac_opts, default=["All"], key="tt_ac")
    with fc2:
        ctry_opts = ["All"] + sorted(ev["_CNT"].unique().tolist())
        pick_cty = st.multiselect("Country", options=ctry_opts, default=["All"], key="tt_cty")
    with fc3:
        src_opts = ["All"] + sorted(ev["_SRC"].unique().tolist())
        pick_src = st.multiselect("JetLearn Deal Source", options=src_opts, default=["All"], key="tt_src")

    def _resolve(vals, all_vals):
        return all_vals if ("All" in vals or not vals) else vals

    ac_sel  = _resolve(pick_ac, sorted(ev["_AC"].unique().tolist()))
    cty_sel = _resolve(pick_cty, sorted(ev["_CNT"].unique().tolist()))
    src_sel = _resolve(pick_src, sorted(ev["_SRC"].unique().tolist()))

    ev = ev[ev["_AC"].isin(ac_sel) & ev["_CNT"].isin(cty_sel) & ev["_SRC"].isin(src_sel)].copy()

    st.caption(f"Window: **{start_d} → {end_d}** • Mode: **{mode}**  • Rows: **{len(ev)}**")

    if ev.empty:
        st.info("No events after applying filters."); return

    # Granularity columns
    ev["_day"] = ev["_when"].dt.date
    ev["_mon"] = _pd.to_datetime(ev["_when"].dt.to_period("M").astype(str))

    # Map dims to columns
    def _map_dim(x):
        return {"Academic Counsellor":"_AC", "Country":"_CNT", "JetLearn Deal Source":"_SRC"}.get(x)

    dim_cols = [_map_dim(x) for x in dims if _map_dim(x)]
    if not dim_cols:
        ev["_All"] = "All"; dim_cols = ["_All"]

    # Build trend aggregation
    if gran == "Daily":
        grp = ["_day"] + dim_cols + ["Metric"]
        ser = ev.groupby(grp, dropna=False).size().rename("Count").reset_index()
        x_enc = _alt.X("_day:T", title=None)
        rename_time_col = {"_day":"Date"}
    else:
        grp = ["_mon"] + dim_cols + ["Metric"]
        ser = ev.groupby(grp, dropna=False).size().rename("Count").reset_index()
        x_enc = _alt.X("_mon:T", title=None)
        rename_time_col = {"_mon":"Month"}

    # Chart selector
    chart_type = st.radio("Chart", ["Stacked Bar","Line"], index=0, horizontal=True, key="tt_chart")

    if chart_type == "Stacked Bar":
        # Stack Metrics by color, x=time (and aggregate across selected dim combination)
        color_enc = _alt.Color("Metric:N", title="Metric")
        ch = _alt.Chart(ser).mark_bar().encode(
            x=x_enc,
            y=_alt.Y("Count:Q"),
            color=color_enc,
            tooltip=[_alt.Tooltip("Metric:N"), _alt.Tooltip("Count:Q")]
        ).properties(height=320)
    else:
        color_enc = _alt.Color("Metric:N", title="Metric")
        ch = _alt.Chart(ser).mark_line(point=True).encode(
            x=x_enc,
            y=_alt.Y("Count:Q"),
            color=color_enc,
            tooltip=[_alt.Tooltip("Metric:N"), _alt.Tooltip("Count:Q")]
        ).properties(height=320)
    st.altair_chart(ch, use_container_width=True)

    # Table for trend
    pretty = ser.rename(columns={**rename_time_col, "_AC":"Academic Counsellor","_CNT":"Country","_SRC":"JetLearn Deal Source"})
    st.dataframe(pretty, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Trial Trend", data=pretty.to_csv(index=False).encode("utf-8"),
                       file_name="trial_trend.csv", mime="text/csv", key="tt_dl_tbl")

    # ---- Percentage trend between two selected metrics
    st.markdown("### % Trend (B / A)")
    metric_opts = ["Lead","Trial","Trial Done","Enrollment"]
    cpa, cpb = st.columns(2)
    with cpa:
        den = st.selectbox("Metric A (denominator)", options=metric_opts, index=1, key="tt_den")
    with cpb:
        num = st.selectbox("Metric B (numerator)", options=metric_opts, index=2, key="tt_num")

    # Build aligned series per time bucket
    if gran == "Daily":
        time_col = "_day"
    else:
        time_col = "_mon"

    pivot = ser.pivot_table(index=time_col, columns="Metric", values="Count", aggfunc="sum", fill_value=0).reset_index()
    if den not in pivot.columns or num not in pivot.columns:
        st.info("Selected metrics have no data in the window."); return

    pivot["Rate"] = _np.where(pivot[den] > 0, pivot[num] / pivot[den], _np.nan)
    # Chart
    ch_rate = _alt.Chart(pivot).mark_line(point=True).encode(
        x=_alt.X(f"{time_col}:T", title=None),
        y=_alt.Y("Rate:Q", axis=_alt.Axis(format="%"), title=f"{num} / {den}"),
        tooltip=[_alt.Tooltip(f"{time_col}:T", title="Time"),
                 _alt.Tooltip(f"{den}:Q"), _alt.Tooltip(f"{num}:Q"),
                 _alt.Tooltip("Rate:Q", format=".2%")]
    ).properties(height=300)
    st.altair_chart(ch_rate, use_container_width=True)

    # Table
    pretty_rate = pivot.rename(columns={time_col: ("Date" if gran=="Daily" else "Month")})
    st.dataframe(pretty_rate, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — % Trend (B over A)", data=pretty_rate.to_csv(index=False).encode("utf-8"),
                       file_name="trial_percentage_trend.csv", mime="text/csv", key="tt_dl_rate")

# Router hook
try:
    if view == "Trial Trend":
        _render_funnel_trial_trend(
            df,
            first_trial_col_hint=None,
            resched_col_hint=None,
            trial_done_col_hint=None,
            enrol_col_hint=None,
            create_col_hint=locals().get("create_col", None),
            counsellor_col_hint=locals().get("counsellor_col", None),
            country_col_hint=locals().get("country_col", None),
            source_col_hint=locals().get("source_col", None),
        )
except Exception as _e:
    import streamlit as st
    st.error(f"Trial Trend failed: {type(_e).__name__}: {_e}")
# =============  /TRIAL TREND  =============


# ================================
# Marketing ▶ referral_Sibling
# ================================
def _render_marketing_referral_sibling(df):
    import streamlit as st
    import pandas as pd
    from datetime import date, timedelta
    st.subheader("Marketing — referral_Sibling")

    if df is None or getattr(df, "empty", True):
        st.info("No data available."); return

    sibling_col = find_col(df, ["Sibling Deal", "Sibling deal", "Sibling_Deal", "SiblingDeal"])
    dealstage_col = find_col(df, ["Deal Stage", "Deal stage", "deal stage", "DEAL STAGE"])
    create_col = find_col(df, ["Create Date", "Deal Create Date", "Created Date", "create date"])

    if not sibling_col:
        st.error("Sibling Deal column not found."); return
    if not dealstage_col:
        st.error("Deal Stage column not found."); return
    if not create_col:
        st.error("Create Date column not found."); return

    # Parse create dates (day-first safe)
    s_create = coerce_datetime(df[create_col])
    d = df.copy()
    d[create_col] = s_create

    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range (by Create Date)", presets, index=2, horizontal=True, key="refsib_rng")

    def _month_bounds(d0: date):
        from calendar import monthrange
        return date(d0.year, d0.month, 1), date(d0.year, d0.month, monthrange(d0.year, d0.month)[1])

    if preset == "Today":
        start_date = today
        end_date = today
    elif preset == "Yesterday":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif preset == "This Month":
        start_date, end_date = month_bounds(today) if 'month_bounds' in globals() else _month_bounds(today)
    elif preset == "Last Month":
        if 'last_month_bounds' in globals():
            start_date, end_date = last_month_bounds(today)
        else:
            first_this = date(today.year, today.month, 1)
            last_prev = first_this - timedelta(days=1)
            start_date, end_date = _month_bounds(last_prev)
    else:
        start_date, end_date = st.date_input("Choose range", value=(today.replace(day=1), today), key="refsib_custom")
        if isinstance(start_date, tuple) or isinstance(end_date, tuple):
            # Streamlit sometimes returns tuple on older builds; guard
            start_date, end_date = start_date[0], end_date[-1]

    # Filter by Create Date
    mask = d[create_col].notna() & d[create_col].dt.date.between(start_date, end_date)
    d = d.loc[mask, [sibling_col, dealstage_col]].copy()

    # Clean labels
    d[sibling_col] = d[sibling_col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    d[dealstage_col] = d[dealstage_col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})

    # Pivot: rows = Sibling Deal values; columns = Deal Stage; values = counts
    pivot = pd.crosstab(index=d[sibling_col], columns=d[dealstage_col])
    pivot = pivot.sort_index()

    st.dataframe(pivot, use_container_width=True)

    st.download_button(
        "Download CSV — referral_Sibling pivot",
        data=pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="marketing_referral_sibling_pivot.csv",
        mime="text/csv",
        key="dl_refsib_csv"
    )


# ---- Dispatcher for Marketing ▸ referral_Sibling
try:
    if st.session_state.get('nav_master') == "Marketing" and st.session_state.get('nav_sub') == "referral_Sibling":
        _render_marketing_referral_sibling(df_f)
except Exception as _e:
    import streamlit as _st
    _st.error(f"referral_Sibling error: {_e}")


# Safety redirect if a removed pill is selected
try:
    if st.session_state.get('nav_master') == "Performance" and st.session_state.get('nav_sub') in ("Lead mix","Lead Mix","lead mix"):
        first_perf = MASTER_SECTIONS.get("Performance", [None])[0]
        if first_perf:
            st.session_state['nav_sub'] = first_perf
            st.rerun()
except Exception:
    pass



# -------------------- Marketing • Deal Score Trend --------------------
def _render_marketing_deal_score_trend(df):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt

    st.subheader("Deal Score Trend")
    st.caption("Explore trends for **New Deal Score**, **Engagement**, **Fit**, and **Threshold** grouped by Country, Source, or AC.")

    # Column normalization
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    date_col = None
    for c in ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        st.error("No valid create date column found.")
        return

    score_cols_map = {
        "New Deal Score": "New Deal Score",
        "Engagement": "New Deal Score engagement",
        "Fit": "New Deal Score fit",
        "Threshold": "New Deal Score threshold",
    }
    # Check presence
    missing = [v for v in score_cols_map.values() if v not in df.columns]
    if missing:
        st.warning("Missing columns: " + ", ".join(missing))
    # Coerce types
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    for v in score_cols_map.values():
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")

    # Filters UI
    c1, c2, c3, c4 = st.columns([1.2,1,1,1.2])
    with c1:
        vars_sel = st.multiselect(
            "Variables",
            ["New Deal Score","Engagement","Fit","Threshold"],
            default=["New Deal Score"],
            help="Choose one or more variables to trend."
        )
    with c2:
        group_by = st.selectbox(
            "Group by",
            ["Country","JetLearn Deal Source","Student/Academic Counsellor","(Overall)"],
            index=0
        )
    with c3:
        gran = st.selectbox("Granularity", ["Daily","Weekly","Monthly"], index=2)
    with c4:
        preset = st.selectbox("Date range",
                              ["This Month","Last Month","Last 90 days","Year to date","Custom"],
                              index=0)

    # Date presets
    today = pd.Timestamp.today().normalize()
    first_day_this_month = today.replace(day=1)
    if preset == "This Month":
        start, end = first_day_this_month, today
    elif preset == "Last Month":
        start = (first_day_this_month - pd.offsets.MonthBegin(1))
        end = first_day_this_month - pd.Timedelta(days=1)
    elif preset == "Last 90 days":
        start, end = today - pd.Timedelta(days=89), today
    elif preset == "Year to date":
        start, end = pd.Timestamp(today.year,1,1), today
    else:
        d1 = st.date_input("Start date", value=today - pd.Timedelta(days=29))
        d2 = st.date_input("End date", value=today)
        start, end = pd.to_datetime(d1), pd.to_datetime(d2)

    # Filter by date
    dfx = df[(df[date_col]>=start) & (df[date_col]<=end)].copy()
    if dfx.empty:
        st.info("No rows in selected date range.")
        return

    # Dependent value selection for the chosen group
    if group_by != "(Overall)":
        _group_key = group_by if group_by in dfx.columns else None
        if _group_key is None:
            st.error(f"Group column not found: {group_by}")
            return
        _vals = sorted([str(x) for x in dfx[_group_key].dropna().unique().tolist()])
        if len(_vals) == 0:
            _vals = ["(Blank)"]
        selected_values = st.multiselect(f"Select {group_by} values", _vals, default=_vals)
        if selected_values:
            dfx = dfx[dfx[_group_key].astype(str).isin(selected_values)]
        else:
            st.info("No values selected — showing nothing for this grouping.")
            dfx = dfx.iloc[0:0]

    # ---- Slider filter over a chosen score variable ----
    slider_candidates = [lbl for lbl in ["New Deal Score","Engagement","Fit","Threshold"] if score_cols_map[lbl] in dfx.columns]
    if slider_candidates:
        csl1, csl2 = st.columns([1.2, 1.2])
        with csl1:
            slider_var_label = st.selectbox("Slider variable", slider_candidates, index=(slider_candidates.index("New Deal Score") if "New Deal Score" in slider_candidates else 0),
                                            help="Filter rows by a numeric range on the selected score variable.")
        slider_var_col = score_cols_map[slider_var_label]
        # Compute min/max from the current window (after date & group filters)
        _series = pd.to_numeric(dfx[slider_var_col], errors="coerce")
        _min = float(np.nanmin(_series)) if _series.notna().any() else 0.0
        _max = float(np.nanmax(_series)) if _series.notna().any() else 0.0
        with csl2:
            step = float(max((_max - _min)/100.0, 0.1))
            rng = st.slider(f"{slider_var_label} range", min_value=float(_min), max_value=float(_max), value=(float(_min), float(_max)), step=step)
        # Apply filter
        dfx = dfx[_series.between(rng[0], rng[1], inclusive="both")]
    else:
        slider_var_label = None

    # KPI box: show total deals after slider & date/group filters
    st.metric("Deals (after filters)", int(len(dfx)))

    # Time key
    if gran == "Daily":
        dfx["_t"] = dfx[date_col].dt.date
    elif gran == "Weekly":
        dfx["_t"] = dfx[date_col].dt.to_period("W-MON").dt.start_time.dt.date
    else:
        dfx["_t"] = dfx[date_col].dt.to_period("M").dt.to_timestamp().dt.date

    # Group key
    if group_by == "(Overall)":
        dfx["_g"] = "Overall"
    else:
        dfx["_g"] = dfx[group_by].fillna("(Blank)").astype(str)

    # Build long frame
    long_frames = []
    for label in vars_sel:
        col = score_cols_map[label]
        if col not in dfx.columns:
            continue
        grp = dfx.groupby(["_t","_g"], as_index=False)[col].agg(np.nanmean)
        grp["Variable"] = label
        grp.rename(columns={col:"Value"}, inplace=True)
        long_frames.append(grp)

    if not long_frames:
        st.warning("No selected variables available in data.")
        return

    out = (pd.concat(long_frames, ignore_index=True)
             .sort_values(["_t","_g","Variable"]))

    # Chart
    base = alt.Chart(out).mark_line(point=True).encode(
        x=alt.X("_t:T", title="Date"),
        y=alt.Y("Value:Q", title="Mean value"),
        color=alt.Color("Variable:N"),
        tooltip=["_t:T","_g:N","Variable:N","Value:Q"]
    )
    chart = base.encode(row=alt.Row("_g:N")) if group_by != "(Overall)" else base
    st.altair_chart(chart, use_container_width=True)

    # Table + CSV
    st.markdown("#### Data")
    st.dataframe(out.rename(columns={"_t":"Date","_g":group_by}), use_container_width=True)
    st.download_button(
        "Download CSV — Deal Score Trend",
        out.rename(columns={"_t":"Date","_g":group_by}).to_csv(index=False).encode("utf-8"),
        "deal_score_trend.csv",
        "text/csv"
    )

# Dispatch
try:
    if master == "Marketing" and st.session_state.get('nav_sub') == "Deal Score Trend":
        _df_any = None
        for _nm in ("dff","df_f","df"):
            _cand = globals().get(_nm, None)
            if _cand is not None:
                _df_any = _cand
                break
        if _df_any is not None:
            _render_marketing_deal_score_trend(_df_any)
        else:
            st.error("No dataframe available to render Deal Score Trend.")
except Exception as _e:
    import traceback, streamlit as st
    st.exception(_e)



# -------------------- Marketing • Deal Score Threshold --------------------
def _render_marketing_deal_score_threshold(df):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt

    st.subheader("Deal Score Threshold")
    st.caption("Trend the counts of deals by **New Deal Score threshold** (e.g., A1, A2...) with grouping, granularity, and MTD/Cohort.")

    # Normalize headers
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    thr_col = "New Deal Score threshold"
    if thr_col not in df.columns:
        st.error(f"Column not found: {thr_col}")
        return

    date_options = [c for c in ["Payment Received Date", "Last Activity Date", "Create Date"] if c in df.columns]
    if not date_options:
        st.error("No suitable date column found (need one of Payment Received Date, Last Activity Date, Create Date).")
        return

    for dc in set(date_options + ["Create Date"]):
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce", dayfirst=True)

    c1, c2, c3, c4 = st.columns([1.2,1,1,1.2])
    with c1:
        event_date = st.selectbox("Event date basis", date_options, index=(date_options.index("Create Date") if "Create Date" in date_options else 0))
    with c2:
        group_by = st.selectbox("Group by", ["Country","JetLearn Deal Source","Student/Academic Counsellor","(Overall)"], index=0)
    with c3:
        gran = st.selectbox("Granularity", ["Daily","Weekly","Monthly"], index=2)
    with c4:
        mode = st.radio("Mode", ["MTD","Cohort"], horizontal=True, index=0)

    c5, c6 = st.columns([1,1])
    with c5:
        preset = st.selectbox("Date range", ["This Month","Last Month","Last 90 days","Year to date","Custom"], index=0)
    with c6:
        chart_type = st.selectbox("Chart type", ["Bar","Stacked Bar","Line"], index=0)

    today = pd.Timestamp.today().normalize()
    first_day_this_month = today.replace(day=1)
    if preset == "This Month":
        start, end = first_day_this_month, today
    elif preset == "Last Month":
        start = (first_day_this_month - pd.offsets.MonthBegin(1))
        end = first_day_this_month - pd.Timedelta(days=1)
    elif preset == "Last 90 days":
        start, end = today - pd.Timedelta(days=89), today
    elif preset == "Year to date":
        start, end = pd.Timestamp(today.year,1,1), today
    else:
        d1 = st.date_input("Start date", value=today - pd.Timedelta(days=29))
        d2 = st.date_input("End date", value=today)
        start, end = pd.to_datetime(d1), pd.to_datetime(d2)

    dfx = df.copy()
    if mode == "MTD":
        dfx = dfx[(dfx[event_date]>=start) & (dfx[event_date]<=end)]
        if "Create Date" in dfx.columns:
            dfx = dfx[(dfx["Create Date"]>=start) & (dfx["Create Date"]<=end)]
    else:
        dfx = dfx[(dfx[event_date]>=start) & (dfx[event_date]<=end)]

    if dfx.empty:
        st.info("No rows in selected date range.")
        return

    if group_by != "(Overall)":
        _group_key = group_by if group_by in dfx.columns else None
        if _group_key is None:
            st.error(f"Group column not found: {group_by}")
            return
        _vals = sorted([str(x) for x in dfx[_group_key].dropna().unique().tolist()])
        if len(_vals) == 0:
            _vals = ["(Blank)"]
        selected_values = st.multiselect(f"Select {group_by} values", _vals, default=_vals)
        if selected_values:
            dfx = dfx[dfx[_group_key].astype(str).isin(selected_values)]
        else:
            st.info("No values selected — showing nothing for this grouping.")
            dfx = dfx.iloc[0:0]

    cats_all = [str(x) if pd.notna(x) else "(Blank)" for x in sorted(dfx[thr_col].dropna().unique())]
    if not cats_all:
        cats_all = ["(Blank)"]
    cats_sel = st.multiselect("Threshold classes", cats_all, default=cats_all)

    if gran == "Daily":
        dfx["_t"] = dfx[event_date].dt.date
    elif gran == "Weekly":
        dfx["_t"] = dfx[event_date].dt.to_period("W-MON").dt.start_time.dt.date
    else:
        dfx["_t"] = dfx[event_date].dt.to_period("M").dt.to_timestamp().dt.date

    if group_by == "(Overall)":
        dfx["_g"] = "Overall"
    else:
        dfx["_g"] = dfx[group_by].fillna("(Blank)").astype(str)

    dfx["_thr"] = dfx[thr_col].fillna("(Blank)").astype(str)
    if cats_sel:
        dfx = dfx[dfx["_thr"].isin(cats_sel)]
    else:
        st.info("No threshold classes selected — showing nothing.")
        dfx = dfx.iloc[0:0]

    if dfx.empty:
        st.info("No data after filters.")
        return

    grp = dfx.groupby(["_t","_g","_thr"], as_index=False).size().rename(columns={"size":"Count"})
    grp = grp.sort_values(["_t","_g","_thr"])

    import altair as alt
    if chart_type == "Line":
        base = alt.Chart(grp).mark_line(point=True)
        enc = base.encode(
            x=alt.X("_t:T", title="Date"),
            y=alt.Y("Count:Q"),
            color=alt.Color("_thr:N", title="Threshold"),
            tooltip=["_t:T","_g:N","_thr:N","Count:Q"]
        )
    elif chart_type == "Stacked Bar":
        base = alt.Chart(grp).mark_bar()
        enc = base.encode(
            x=alt.X("_t:T", title="Date"),
            y=alt.Y("Count:Q", stack="zero"),
            color=alt.Color("_thr:N", title="Threshold"),
            tooltip=["_t:T","_g:N","_thr:N","Count:Q"]
        )
    else:  # Bar
        base = alt.Chart(grp).mark_bar()
        enc = base.encode(
            x=alt.X("_t:T", title="Date"),
            y=alt.Y("Count:Q"),
            color=alt.Color("_thr:N", title="Threshold"),
            tooltip=["_t:T","_g:N","_thr:N","Count:Q"]
        )

    chart = enc.encode(row=alt.Row("_g:N")) if group_by != "(Overall)" else enc
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Data")
    st.dataframe(grp.rename(columns={"_t":"Date","_g":group_by,"_thr":"Threshold"}), use_container_width=True)
    st.download_button(
        "Download CSV — Deal Score Threshold",
        grp.rename(columns={"_t":"Date","_g":group_by,"_thr":"Threshold"}).to_csv(index=False).encode("utf-8"),
        "deal_score_threshold.csv",
        "text/csv"
    )

# Dispatch
try:
    if master == "Marketing" and st.session_state.get('nav_sub') == "Deal Score Threshold":
        _df_any = None
        for _nm in ("dff","df_f","df"):
            _cand = globals().get(_nm, None)
            if _cand is not None:
                _df_any = _cand
                break
        if _df_any is not None:
            _render_marketing_deal_score_threshold(_df_any)
        else:
            st.error("No dataframe available to render Deal Score Threshold.")
except Exception as _e:
    import traceback, streamlit as st
    st.exception(_e)



# -------------------- Marketing • Invalid Deals --------------------
def _render_marketing_invalid_deals(df):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import altair as alt
    import re as _re_mod

    st.subheader("Invalid Deals")
    st.caption("Counts and distribution for **Deal Stage** values (defaults to stages containing 'invalid').")

    # Use RAW data only for this pill (bypass global 'exclude invalids')
    import streamlit as st, pandas as pd
    @st.cache_data(show_spinner=False)
    def _load_raw_df(_path):
        _df0 = pd.read_csv(_path)
        _df0.columns = [c.strip() for c in _df0.columns]
        return _df0
    _data_path = st.session_state.get("data_src") or globals().get("DEFAULT_DATA_PATH", "Master_sheet-DB.csv")
    df_in = _load_raw_df(_data_path).copy()

    deal_stage_col = None
    for c in ["Deal Stage","Stage","_stage","Pipeline Stage"]:
        if c in df_in.columns:
            deal_stage_col = c
            break
    if deal_stage_col is None:
        st.error("No Deal Stage column found.")
        return

    date_options = [c for c in ["Payment Received Date", "Last Activity Date", "Create Date"] if c in df_in.columns]
    if not date_options:
        st.error("No suitable date column found (need one of Payment Received Date, Last Activity Date, Create Date).")
        return

    for dc in set(date_options + ["Create Date"]):
        if dc in df_in.columns:
            df_in[dc] = pd.to_datetime(df_in[dc], errors="coerce", dayfirst=True)

    c1, c2, c3, c4 = st.columns([1.2,1,1,1.2])
    with c1:
        event_date = st.selectbox(
            "Event date basis",
            date_options,
            index=(date_options.index("Create Date") if "Create Date" in date_options else 0),
            help="This date drives the trend window for MTD/Cohort."
        )
    with c2:
        group_by = st.selectbox("Group by", ["Country","JetLearn Deal Source","Student/Academic Counsellor","(Overall)"], index=0)
    with c3:
        gran = st.selectbox("Granularity", ["Daily","Weekly","Monthly"], index=2)
    with c4:
        mode = st.radio("Mode", ["MTD","Cohort"], horizontal=True, index=0)

    c5, c6 = st.columns([1,1])
    with c5:
        preset = st.selectbox("Date range", ["This Month","Last Month","Last 90 days","Year to date","Custom"], index=0)
    with c6:
        chart_type = st.selectbox("Chart type", ["Bar","Stacked Bar","Line"], index=0)

    today = pd.Timestamp.today().normalize()
    first_day_this_month = today.replace(day=1)
    if preset == "This Month":
        start, end = first_day_this_month, today
    elif preset == "Last Month":
        start = (first_day_this_month - pd.offsets.MonthBegin(1))
        end = first_day_this_month - pd.Timedelta(days=1)
    elif preset == "Last 90 days":
        start, end = today - pd.Timedelta(days=89), today
    elif preset == "Year to date":
        start, end = pd.Timestamp(today.year,1,1), today
    else:
        d1 = st.date_input("Start date", value=today - pd.Timedelta(days=29))
        d2 = st.date_input("End date", value=today)
        start, end = pd.to_datetime(d1), pd.to_datetime(d2)

    # Build working window dfw
    dfw = df_in.copy()
    if mode == "MTD":
        dfw = dfw[(dfw[event_date]>=start) & (dfw[event_date]<=end)]
        if "Create Date" in dfw.columns:
            dfw = dfw[(dfw["Create Date"]>=start) & (dfw["Create Date"]<=end)]
    else:
        dfw = dfw[(dfw[event_date]>=start) & (dfw[event_date]<=end)]

    if dfw.empty:
        st.info("No rows in selected date range.")
        return

    # Dependent group value select
    if group_by != "(Overall)":
        _group_key = group_by if group_by in dfw.columns else None
        if _group_key is None:
            st.error(f"Group column not found: {group_by}")
            return
        _vals = sorted([str(x) for x in dfw[_group_key].dropna().unique().tolist()])
        if len(_vals) == 0:
            _vals = ["(Blank)"]
        selected_values = st.multiselect(f"Select {group_by} values", _vals, default=_vals)
        if selected_values:
            dfw = dfw[dfw[_group_key].astype(str).isin(selected_values)]
        else:
            st.info("No values selected — showing nothing for this grouping.")
            dfw = dfw.iloc[0:0]

    if dfw.empty:
        st.info("No data after group filter.")
        return

    # Deal Stage (dlistage) — ALL stages available in window/group, default to those containing 'invalid'
    _all_stages = sorted([str(x) for x in dfw[deal_stage_col].dropna().unique().tolist()])
    if not _all_stages:
        st.info("No Deal Stage values in the current selection.")
        return
    _default_invalid = [s for s in _all_stages if _re_mod.search(r"invalid", s, flags=_re_mod.IGNORECASE)]
    default_sel = _default_invalid if _default_invalid else _all_stages
    sel_stages = st.multiselect("Deal Stage (dlistage)", _all_stages, default=default_sel,
                                help="Defaults to all stages containing 'invalid'. Change to any subset you want.")
    if sel_stages:
        dfx = dfw[dfw[deal_stage_col].astype(str).isin(sel_stages)].copy()
    else:
        st.info("No stages selected — showing nothing.")
        dfx = dfw.iloc[0:0].copy()

    if dfx.empty:
        st.info("No data after Deal Stage (dlistage) filter.")
        return

    # KPI total
    st.metric("Deals (after filters)", int(len(dfx)))

    # Distribution basis: exact Deal Stage string
    dfx["_stage_label"] = dfx[deal_stage_col].astype(str)

    # Time key
    if gran == "Daily":
        dfx["_t"] = dfx[event_date].dt.date
    elif gran == "Weekly":
        dfx["_t"] = dfx[event_date].dt.to_period("W-MON").dt.start_time.dt.date
    else:
        dfx["_t"] = dfx[event_date].dt.to_period("M").dt.to_timestamp().dt.date

    # Group key
    if group_by == "(Overall)":
        dfx["_g"] = "Overall"
    else:
        dfx["_g"] = dfx[group_by].fillna("(Blank)").astype(str)

    grp = dfx.groupby(["_t","_g","_stage_label"], as_index=False).size().rename(columns={"size":"Count"})
    grp = grp.sort_values(["_t","_g","_stage_label"])

    if chart_type == "Line":
        base = alt.Chart(grp).mark_line(point=True)
        enc = base.encode(
            x=alt.X("_t:T", title="Date"),
            y=alt.Y("Count:Q"),
            color=alt.Color("_stage_label:N", title="Deal Stage"),
            tooltip=["_t:T","_g:N","_stage_label:N","Count:Q"]
        )
    elif chart_type == "Stacked Bar":
        base = alt.Chart(grp).mark_bar()
        enc = base.encode(
            x=alt.X("_t:T", title="Date"),
            y=alt.Y("Count:Q", stack="zero"),
            color=alt.Color("_stage_label:N", title="Deal Stage"),
            tooltip=["_t:T","_g:N","_stage_label:N","Count:Q"]
        )
    else:  # Bar
        base = alt.Chart(grp).mark_bar()
        enc = base.encode(
            x=alt.X("_t:T", title="Date"),
            y=alt.Y("Count:Q"),
            color=alt.Color("_stage_label:N", title="Deal Stage"),
            tooltip=["_t:T","_g:N","_stage_label:N","Count:Q"]
        )

    chart = enc.encode(row=alt.Row("_g:N")) if group_by != "(Overall)" else enc
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Data")
    st.dataframe(grp.rename(columns={"_t":"Date","_g":group_by,"_stage_label":"Deal Stage"}), use_container_width=True)
    st.download_button(
        "Download CSV — Invalid Deals",
        grp.rename(columns={"_t":"Date","_g":group_by,"_stage_label":"Deal Stage"}).to_csv(index=False).encode("utf-8"),
        "invalid_deals.csv",
        "text/csv"
    )

# Dispatch
try:
    if master == "Marketing" and st.session_state.get('nav_sub') == "Invalid Deals":
        _df_any = None
        for _nm in ("df","dff","df_f"):
            _cand = globals().get(_nm, None)
            if _cand is not None:
                _df_any = _cand
                break
        if _df_any is not None:
            _render_marketing_invalid_deals(_df_any)
        else:
            st.error("No dataframe available to render Invalid Deals.")
except Exception as _e:
    import traceback, streamlit as st
    st.exception(_e)
