import os
import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI

# ============================
# 0. 기본 설정 & 스타일
# ============================
st.set_page_config(
    page_title="Sleep Quality App",
    page_icon="🛌",
    layout="wide",
)

# 커스텀 CSS (배경, 카드, 버튼 스타일 등)
st.markdown(
    """
    <style>
    /* 전체 배경 톤 다운 */
    .main {
        background-color: #f4f6fb;
    }

    /* 제목과 설명이 들어가는 상단 헤더 */
    .app-header {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        color: white;
        margin-bottom: 1.2rem;
    }

    .app-header h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.3rem;
    }

    .app-header p {
        margin-top: 0.2rem;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* 카드 스타일 컨테이너 */
    .card {
        background-color: white;
        padding: 1.2rem 1.3rem;
        border-radius: 16px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }

    /* 버튼 둥글게 */
    div.stButton > button {
        border-radius: 999px;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
    }

    /* 탭 제목 살짝 강조 */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* 데이터프레임 테이블 여백 줄이기 */
    .stDataFrame {
        margin-top: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# 1. XGBoost 회귀 모델 로드
# ============================
@st.cache_resource
def load_xgb_model():
    model = joblib.load("sleep_model.pkl")
    feature_cols = joblib.load("sleep_features.pkl")
    return model, list(feature_cols)

model, feature_cols = load_xgb_model()

# ----------------------------
# 1-1. 카페인 컬럼 이름 추론
# ----------------------------
caffeine_col = None
for col in feature_cols:
    if "cafe" in col.lower() or "caffeine" in col.lower():
        caffeine_col = col
        break
# 필요하면 명시적으로 지정 가능
# caffeine_col = "Caffeine_Intake"

# ============================
# 2. 사이드바 (설정 + API Key 입력)
# ============================
with st.sidebar:
    st.markdown("### ⚙️ 설정 & 안내")
    st.write(
        """
        - **입력 방식**: 직접 입력 또는 CSV 업로드  
        - **모델**: XGBoost 회귀 + ChatGPT 리포트  
        - **카페인 변수**: 카페인 계산기를 통해 자동 계산  
        """
    )

    if caffeine_col:
        st.caption(f"카페인 컬럼 감지됨: `{caffeine_col}`")
    else:
        st.caption("카페인 관련 컬럼을 자동으로 찾지 못했습니다. feature_cols를 확인하세요.")

    # 🔑 OpenAI API Key 입력
    api_key = st.text_input("🔑 OpenAI API Key 입력", type="password")
    st.caption("입력한 키는 이 세션 내에서만 사용되며, 코드나 GitHub에 저장되지 않습니다.")

# ============================
# 3. OpenAI Client 헬퍼
# ============================
def get_client():
    """
    사이드바에서 입력한 API Key로 OpenAI 클라이언트 생성.
    """
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

# ============================
# 4. LLM 호출 함수
# ============================
def call_llm(prompt: str) -> str:
    """
    ChatGPT API 호출. 사이드바에 입력된 API Key를 사용.
    """
    client = get_client()

    if client is None:
        return "❌ OpenAI API Key가 입력되지 않았거나 유효하지 않습니다. 왼쪽 설정에서 키를 확인해 주세요."

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 한국어로 의료 리포트를 작성하는 전문가이자 조언자다. "
                        "프롬프트에 주어진 사용자 데이터를 바탕으로만 수면 상태 리포트를 작성하고, "
                        "프롬프트의 지시문 자체를 그대로 옮기거나 요약하지 않는다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ LLM 호출 중 오류가 발생했습니다: {e}"

# ============================
# 5. 리포트용 프롬프트 생성 함수
# ============================
def build_prompt_from_row(row_dict: dict, predicted_score: float) -> str:
    feature_text_lines = [f"- {k}: {v}" for k, v in row_dict.items()]
    feature_text = "\n".join(feature_text_lines)

    prompt = f"""
너는 수면의학(sleep medicine) 전문의를 보조하는 임상 보고서 생성 AI이다.
아래의 “사용자 데이터”를 기반으로, 사용자 맞춤형 임상적 수면 평가 보고서를 작성하라.
보고서에는 의학적 객관성, 근거 기반 표현, 임상적 판단의 논리 구조가 반영되어야 한다.

프롬프트의 지시문을 재작성하거나 요약하지 않는다.
아래에 제시된 네 개의 섹션 외의 내용을 추가하지 않는다.

[사용자 데이터]
{feature_text}
- 예측된 Sleep_Quality_Score: {predicted_score} / 10

[보고서 형식 — 반드시 이 형식만 사용할 것]

1. 종합 수면 상태 평가(Clinical Summary)
   - 수면의 질을 정량적/정성적으로 요약한다.
   - 예측 점수와 실제 입력 패턴의 관계를 임상적 관점에서 해석한다.

2. 수면 저하 위험 요인 분석(Risk Factor Interpretation)
   - 사용자의 입력 데이터 중 수면의 질을 저하시킬 수 있는 요인들을 병태생리학적으로 설명한다.
   - 수면 시간, 기상 시간, 카페인 섭취량, 스트레스 지표, 생활 습관 요인 등이 수면 구조(sleep architecture)에 미치는 영향을 포함한다.
   - 가능하다면 대표적인 연구/임상 가이드라인(예: AASM, Harvard Sleep Health)에서 제시하는 권장치 기준을 참고하듯 서술한다.

3. 근거 기반 개선 전략(Evidence-based Recommendations)
   - 사용자가 실천 가능한 행동을 4~6개 bullet point로 제시한다.
   - 각 항목은 “임상적 근거 또는 생리적 메커니즘 → 실천 전략”의 구조로 작성한다.
   - 예: “카페인 대사 반감기(5–7시간)를 고려하면 오후 2시 이후 섭취 제한 권고”와 같은 식의 전문가 수준 조언을 포함한다.

4. 주의가 필요한 신호 및 전문의 상담 권고(Warning Signs)
   - 입력 데이터를 기반으로 ‘주의가 필요한 패턴’ 또는 ‘수면장애 가능성’을 간단히 언급한다.
   - 불면증(Insomnia), 기면증, 수면무호흡증, 일주기리듬 수면장애 등의 가능성과 관련된 징후가 있으면 조건부로 언급한다.
   - 다만 진단이나 확정 표현은 절대 하지 말고 “가능성 있음”, “추가 평가 필요” 수준으로 작성한다.

반드시 위 4개 섹션만 포함해 한국어로 작성하라.
프롬프트 지시문을 반복하거나 재작성하지 말고, 사용자 데이터 기반의 전문 임상 보고서만 출력하라.
"""
    return prompt

# ============================
# 6. 카페인 계산기 UI
# ============================
def caffeine_calculator_ui():
    st.markdown("#### ☕ 카페인 계산기")
    st.caption("오늘 마신 음료 개수를 입력하면, 하루 총 카페인 섭취량(mg)을 자동 계산합니다.")

    drinks = {
        "아메리카노(레귤러 1잔)": 120,
        "에스프레소 1샷": 75,
        "캔커피 1캔": 80,
        "에너지 드링크 1캔(250ml)": 80,
        "녹차 1잔": 30,
        "홍차 1잔": 40,
        "콜라 1캔(355ml)": 35,
        "디카페인 커피 1잔": 5,
    }

    total_caffeine = 0
    cols = st.columns(2)
    items = list(drinks.items())
    half = (len(items) + 1) // 2

    for i, (name, mg) in enumerate(items):
        with cols[0 if i < half else 1]:
            cnt = st.number_input(
                f"{name} (약 {mg} mg / 1개)",
                min_value=0,
                step=1,
                key=f"caf_{i}",
            )
        total_caffeine += cnt * mg

    st.info(f"오늘 총 카페인 섭취량: **{total_caffeine} mg** (모델 입력값으로 사용됩니다)")
    return float(total_caffeine)

# ============================
# 7. 상단 헤더
# ============================
st.markdown(
    """
    <div class="app-header">
        <h1>🛌 Sleep Quality Report Generator</h1>
        <p>XGBoost로 수면의 질 점수를 예측하고, ChatGPT로 개인 맞춤형 수면 리포트를 생성합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================
# 8. 메인 탭 UI
# ============================
tab_manual, tab_csv = st.tabs(["✍️ 직접 입력", "📂 CSV 업로드"])

# ----------------------------
# 탭 1: 직접 입력
# ----------------------------
with tab_manual:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✍️ 한 명의 데이터 직접 입력")

    st.caption("각 변수 값을 입력한 뒤, 하단의 버튼을 눌러 예측 및 리포트를 생성하세요.")

    # 좌/우 영역: 왼쪽 입력, 오른쪽 결과
    col_left, col_right = st.columns([1.1, 1.1])

    with col_left:
        st.markdown("#### 📥 입력 값")

        input_data = {}

        # 1) 카페인 컬럼 처리
        if caffeine_col is not None:
            with st.expander("☕ 카페인 섭취량 계산기로 자동 입력하기", expanded=True):
                caffeine_value = caffeine_calculator_ui()
            input_data[caffeine_col] = caffeine_value

        # 2) 나머지 변수들 입력
        for col in feature_cols:
            if col == caffeine_col:
                continue
            input_data[col] = st.number_input(label=col, value=0.0)

        do_predict = st.button("🧮 예측 및 리포트 생성", key="manual_predict")

    with col_right:
        st.markdown("#### 📊 예측 결과 & 리포트")

        if do_predict:
            # (1) OpenAI 키 체크
            if not api_key:
                st.error("먼저 왼쪽 사이드바에서 OpenAI API Key를 입력해 주세요.")
            else:
                # (1) XGBoost 예측
                input_df = pd.DataFrame([input_data])
                input_df = input_df[feature_cols]

                predicted_score = float(model.predict(input_df)[0])
                st.success(f"예측된 Sleep_Quality_Score: **{predicted_score:.2f}** / 10")

                # (2) LLM 리포트
                prompt = build_prompt_from_row(input_data, predicted_score)

                with st.spinner("ChatGPT가 리포트를 작성 중입니다..."):
                    report = call_llm(prompt)

                st.markdown("##### 📄 자동 생성 리포트")
                st.markdown(report)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 탭 2: CSV 업로드
# ----------------------------
with tab_csv:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 CSV로 여러 날짜 데이터 업로드")

    st.caption(
        f"""
        - CSV에는 최소한 다음 컬럼들이 포함되어야 합니다:  
          `{', '.join(feature_cols)}`
        - CSV에서는 카페인 컬럼(`{caffeine_col}`) 값도 미리 숫자(mg 등)로 계산해 넣어두세요.
        """
    )

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

    if uploaded_file is not None:
        try:
            df_csv = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
            df_csv = None

        if df_csv is not None:
            st.markdown("#### 🔍 업로드한 원본 데이터")
            st.dataframe(df_csv, use_container_width=True)

            missing_cols = [c for c in feature_cols if c not in df_csv.columns]
            if missing_cols:
                st.error(f"다음 컬럼이 CSV에 없습니다: {missing_cols}")
            else:
                X_csv = df_csv[feature_cols]
                preds = model.predict(X_csv)
                df_result = df_csv.copy()
                df_result["Predicted_Sleep_Quality_Score"] = preds

                st.markdown("#### 🔢 예측이 완료된 데이터")
                st.dataframe(df_result, use_container_width=True)

                st.markdown("#### 📄 특정 행을 선택해 리포트 생성")

                idx_options = list(df_result.index)
                selected_idx = st.selectbox(
                    "리포트를 생성할 행의 인덱스를 선택하세요 (좌측 테이블 index 참고)",
                    idx_options,
                )

                if st.button("선택한 행으로 리포트 생성", key="csv_report"):
                    if not api_key:
                        st.error("먼저 왼쪽 사이드바에서 OpenAI API Key를 입력해 주세요.")
                    else:
                        row = df_result.loc[selected_idx, :]
                        row_features = {col: row[col] for col in feature_cols}
                        predicted_score_row = float(row["Predicted_Sleep_Quality_Score"])

                        st.success(
                            f"선택한 행의 예측 Sleep_Quality_Score: **{predicted_score_row:.2f}** / 10"
                        )

                        prompt = build_prompt_from_row(row_features, predicted_score_row)

                        with st.spinner("ChatGPT가 리포트를 작성 중입니다..."):
                            report = call_llm(prompt)

                        st.markdown("##### 📄 자동 생성 리포트")
                        st.markdown(report)

    st.markdown('</div>', unsafe_allow_html=True)