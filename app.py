
            with st.expander("ข้อมูลเทคนิคเพิ่มเติม (ไม่ต้องแก้ก็ได้)"):
                st.caption("เป็นรหัสตัวเลขตามชุดข้อมูลต้นแบบ ใช้ค่าเริ่มต้นได้")
                c1, c2, c3, c4 = st.columns(4)
                os_ = c1.number_input("ระบบปฏิบัติการ (1-8)", 1, 8, step=1, key="os")
                browser = c2.number_input("เบราว์เซอร์ (1-13)", 1, 13, step=1, key="browser")
                region = c3.number_input("ภูมิภาค (1-9)", 1, 9, step=1, key="region")
                traffic = c4.number_input("ช่องทางเข้าเว็บ (1-20)", 1, 20, step=1, key="traffic")

        submitted = st.form_submit_button("ทำนายผล", **stretch(st.form_submit_button))

    values = {
        "admin": admin, "admin_min": admin_min,
        "info": info, "info_min": info_min,
        "prod": prod, "prod_min": prod_min,
        "bounce_pct": bounce, "exit_pct": exit_,
        "page_value": page_val, "special": special,
        "month": month, "visitor": visitor, "weekend": weekend,
        "os": os_, "browser": browser, "region": region, "traffic": traffic,
    }
    return submitted, values


def render_result(model, values: dict):
    X = build_features(to_model_inputs(values), model.feature_names_in_)
    proba = float(model.predict_proba(X)[0][1])
    title, advice, color = describe_result(proba)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">ค่าความน่าจะเป็นในการซื้อสินค้าตามโมเดล</div>
            <div class="result-value" style="color:{color}">{proba * 100:.1f}%</div>
            <div class="result-text" style="color:{color}">{title}</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{proba * 100:.1f}%; background:{color}"></div></div>
            <div class="result-advice">{advice}</div>
            <div class="result-note">ผลนี้เป็นการประมาณจากโมเดล ไม่ใช่ค่าที่แน่นอน</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    st.markdown("---")
    section(4, "ประสิทธิภาพของโมเดล", "ผลการประเมินโมเดลกับข้อมูลทดสอบที่โมเดลไม่เคยเห็นมาก่อน")

    metrics = fill_metrics_from_confusion(load_metrics())
    items = [
        ("Accuracy", "ความถูกต้องโดยรวม (Accuracy)", metrics.get("accuracy"), True),
        ("Precision", "ความแม่นยำของการทำนายคลาสซื้อ (Precision)", metrics.get("precision"), False),
        ("Recall", "ความสามารถในการตรวจพบคลาสซื้อ (Recall)", metrics.get("recall"), False),
        ("F1-score", "F1-score", metrics.get("f1_score"), False),
        ("ROC-AUC", "ROC-AUC", metrics.get("roc_auc"), False),
    ]
    items = [(n, th, as_percent(v), p) for n, th, v, p in items if v is not None]

    if not items:
        st.info("ยังไม่ได้ระบุค่าประสิทธิภาพของโมเดล — กรอกค่าจากการประเมินโมเดลลงในไฟล์ metrics.json")
        return

    cols = st.columns(len(items))
    for col, (name, th, value, primary) in zip(cols, items):
        css = "metric-card primary" if primary else "metric-card"
        col.markdown(
            f'<div class="{css}"><div class="metric-name">{th}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    note = metrics.get("note")
    st.caption(
        note if note else
        "ค่าทั้งหมดเป็นผลการประเมินโมเดลจากชุดข้อมูลทดสอบ "
        "โดย Accuracy คือสัดส่วนการทำนายที่ถูกต้องทั้งหมด"
    )

    cm = get_confusion(metrics)
    if cm is not None:
        st.markdown('<div class="section-hint" style="margin-top:1rem">ตาราง Confusion Matrix '
                    '(ช่องสีเขียวคือทำนายถูก)</div>', unsafe_allow_html=True)
        st.markdown(confusion_html(cm), unsafe_allow_html=True)


def render_footer():
    st.markdown(
        f'<div class="app-footer">ผู้พัฒนา<br><b>{DEVELOPERS}</b></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🛒", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    for key, val in DEFAULTS.items():
        st.session_state.setdefault(key, val)

    render_header()

    try:
        model = load_model()
    except Exception as e:  # noqa: BLE001
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

    render_presets()
    submitted, values = render_form()
    if submitted:
        render_result(model, values)

    render_metrics()
    render_footer()


if __name__ == "__main__":
    main()
