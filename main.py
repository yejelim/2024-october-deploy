import streamlit as st
import streamlit.components.v1 as components
import openai
import boto3
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from datetime import datetime

# 페이지 설정 (가장 상단에 위치해야 함)
st.set_page_config(
    page_title="의료비 삭감 판정 어시스트 - beta version.",
    layout="wide",  # 창 전체를 사용하도록 설정
)

# 세션 상태 변수 초기화
def initialize_session_state():
    session_state_defaults = {
        'is_clinical_note': False,
        'conversation': [],
        'user_input': '',
        'overall_decision': '',
        'explanations': [],
        'results_displayed': False,
        'score_parsing_attempt': 0,      # 스코어 추출 재시도 횟수
        'embedding_search_attempt': 0,   # 임베딩 및 검색 재시도 횟수
        'max_attempts': 3,               # 최대 재시도 횟수
        'retry_type': None,              # 'score_parsing' 또는 'embedding_search'
        'vectors': [],
        'metadatas': [],
        'full_response': '',
        'scores': {},
        'retry_attempts': 0,
        'upgraded_note': None,
        'copy_text': ''
    }
    for key, value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# OpenAI API 키 설정 (Streamlit secrets 사용)
openai.api_key = st.secrets["openai"]["openai_api_key"]

# 로고 추가 함수
def add_logo():
    with st.sidebar:
        logo_path = "logo.png"  # 올바른 URL 사용
        st.image(logo_path, width=150)  # 로고 크기 조정

# 임상 노트 여부 확인 함수
def check_if_clinical_note(text):
    try:
        prompt = f"다음 텍스트가 임상 노트인지 여부를 판단해주세요:\n\n{text}\n\n이 텍스트는 임상 노트입니까? (예/아니오)"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 임상 문서를 판별하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "예" in answer
    except Exception as e:
        st.error(f"임상 노트 판별 중 오류 발생: {e}")
        return False


# 사용자 로그를 수집하는 함수
def save_user_log_to_s3():
    user_log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": str(uuid.uuid4()),
        "occupation": st.session_state.get("occupation", ""),
        "department": st.session_state.get("department", ""),
        "structured_input": st.session_state.get("structured_input", ""),
        "errors": st.session_state.get("errors", [])
    }
    
    st.session_state['user_log_data'] = user_log_data
    user_log_json = json.dumps(user_log_data, ensure_ascii=False)

    bucket_name = "medinsurance-assist-beta-user-log"
    file_name = f"user_logs/{user_log_data['timestamp']}_{user_log_data['session_id']}.json"

    s3_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["access_key"],
        aws_secret_access_key=st.secrets["aws"]["secret_key"],
        region_name='ap-northeast-2'
    )

    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=user_log_json)
        # 로그 저장 성공 시 별도의 메시지를 표시하지 않아도 됩니다.
    except Exception as e:
        st.error(f"로그 수집 중 오류 발생: {e}")


#예시 임상노트 데이터 사용자가 선택할 수 있게 추가
demo_clinical_notes = {
    "신경외과-사례1": ("신경외과 (Neuro-Surgery)", "왼쪽 종아리가 당긴지 13일 된 환자인데 다른 병원에서 처방받은 약으로 보존적 치료했는데 효과가 없었다. 환자는 내원 당시 엄지발가락의 근력이 4로 저하되어 있었다. 요추 MRI를 본원에서 2023년 7월 14일에 촬영하였고 요추 4-5번간 디스크 파열 및 추간판 탈출로 인한 신경근 압박 소견이 확인 되었다.  근력 저하를 근거로 디스크 제거술을 2023년 7월 15일에 진행하였다."),
    "혈관외과-사례1": ("혈관외과 (Vascular Surgery)", "남/50세. 수술전후 진단: Obstruction of AVBG, Lt.upperarm. 수술명: Open Thrombectomy, Segmental resection of stenosis area, Jump graft- Anesthesia:General. Op Finding-1) GVA stenosis에 의한 폐쇄로 보임. 2) GVA 상방 Axillary vein에 new graft 연결. 3) Upperarm straight graft임."),
    "대장항문외과-사례1": ("대장항문외과 (Colorectal Surgery)", "대장항문외과 환자의 임상 노트 예시입니다."),
    "정맥경장영양-사례1": ("정맥경장영양 (TPN)", "남/49세, 9일 전 입원. 8일 전 췌장암 두부 절제 후 단백아미노제재 TPN 1일 1회, 총 4회 투여.")
}

# 콜백을 사용해서 selectbox 예시노트 선택시 자동으로 text_area, department 업데이트를 UI로 반영해주는 함수
def update_example_note():
    selected_example = st.session_state['selected_example']
    if selected_example != "없음":
        department, example_note = demo_clinical_notes[selected_example]
        st.session_state['user_input'] = example_note
        st.session_state['department'] = department

# 사용자 정보 및 입력을 수집하는 함수
def collect_user_input():    
    # 예시 임상노트를 선택하는 경우에 대한 부분 추가
    st.subheader("예시 임상노트 선택")
    st.selectbox(
        "아래에서 예시 임상노트를 선택하세요:",
        ["없음"] + list(demo_clinical_notes.keys()),
        key="selected_example",
        on_change=update_example_note
    )

    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    user_input = st.text_area(
        "",
        height=500,
        value=st.session_state.get('user_input', ""),
        placeholder="SOAP 등의 임상기록 및 치료 방법 (약물, 시술, 수술) 등을 입력해주세요.",
        key='user_input'
    )

    if user_input != st.session_state['user_input']:
        st.session_state['user_input'] = user_input

    if user_input:
        with st.spinner("임상 노트 여부 확인 중..."):
            is_clinical_note = check_if_clinical_note(user_input)
        if not is_clinical_note:
            st.warning("입력한 텍스트가 임상노트가 아닙니다. 텍스트를 확인해주세요.")
        else:
            st.session_state.is_clinical_note = True

    st.subheader("어떤 분야에 종사하시나요?")
    occupation = st.radio(
        "직업을 선택하세요:",
        options=["의사", "간호사", "병원내 청구팀", "기타"],
        index=0
    )

    if occupation == "기타":
        other_occupation = st.text_input("직업을 입력해주세요:")
    else:
        other_occupation = None

    # Department 초기화 확인 및 선택창
    if 'department' not in st.session_state:
        st.session_state['department'] = ""

    department_options = [
        "신경외과 (Neuro-Surgery)",
        "혈관외과 (Vascular Surgery)",
        "대장항문외과 (Colorectal Surgery)",
        "정맥경장영양 (TPN)"
    ]

    st.subheader("어떤 분과에 재직 중인지 알려주세요.")
    department = st.selectbox(
        "분과를 선택하세요:",
        options=department_options,
        index=0 if st.session_state['department'] == "" else department_options.index(st.session_state['department']),
        key='department'  # 세션 상태와 연동
    )

    # 세션 상태에 사용자 정보 저장
    st.session_state['occupation'] = occupation
    st.session_state['other_occupation'] = other_occupation

    # 체크박스 크기 조절을 위한 CSS
    st.markdown("""
    <style>
    /* 체크박스 크기 증가 */
    [data-testid="stCheckbox"] > label > div:first-child {
        transform: scale(1.5);
    }
        
    /* 빨간색 안내 문구와 글자 크기 조정 */
    .warning-text {
        color: red;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 체크박스 위에 빨간색 안내 문구 추가 (체크되지 않은 경우에만 표시)
    if not st.session_state.get('agree_to_collect', False):
        st.markdown('<div class="warning-text">약관에 동의하셔야 삭감여부 확인이 가능합니다.</div>', unsafe_allow_html=True)

    agree_to_collect = st.checkbox(
        "사용자 정보를 수집하는 것에 동의합니다. 사용자의 텍스트 입력은 개인정보 보호를 위해 수집되지 않으며, 수집된 정보는 일정 기간 후 파기됩니다.",
        key="agree_to_collect"
    )

    # '삭감 여부 확인' 버튼을 체크박스 동의 여부에 따라 활성화/비활성화
    st.session_state['button_disabled'] = not agree_to_collect

    return occupation, other_occupation, department, user_input


# 분과 데이터셋: 추가될 때마다 업데이트 할 부분
department_datasets = {
    "신경외과 (Neuro-Surgery)": {
        "bucket_name": "hemochat-rag-database",
        "file_key": "18_aga_tagged_embedded_data.json"
    },
    "혈관외과 (Vascular Surgery)": {
        "bucket_name": "hemochat-rag-database",
        "file_key": "tagged_vascular_general_criterion_fixed.json"
    },
    "대장항문외과 (Colorectal Surgery)": {
        "bucket_name": "hemochat-rag-database",
        "file_key": "tagged_colorectal_general_criterion_fixed.json"
    },
    "정맥경장영양 (TPN)": {
        "bucket_name": "hemochat-rag-database",
        "file_key": "Experimental_title_only_embedded_TPN_criterion.json"
    }
}

# 선택된 분과에 따라 해당 데이터셋을 로드하는 함수
@st.cache_data
def load_data_if_department_selected(department):
    if department in department_datasets:
        dataset_info = department_datasets[department]
        bucket_name = dataset_info["bucket_name"]
        file_key = dataset_info["file_key"]

        st.write(f"{department} 데이터 로드 중...")
        try:
            embedded_data = load_data_from_s3(bucket_name, file_key)
            # st.success("데이터 로드 완료.")
            return embedded_data
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            return []
    else:
        st.warning(f"현재 {department}에 대한 데이터셋은 준비 중입니다.")
        return []

# AWS S3에서 임베딩 데이터를 로드하는 함수
def load_data_from_s3(bucket_name, file_key):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["access_key"],
            aws_secret_access_key=st.secrets["aws"]["secret_key"],
            region_name='ap-northeast-2'
        )
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read().decode('utf-8')
        return json.loads(data)
    except Exception as e:
        st.error(f"S3에서 데이터 로드 중 오류 발생: {e}")
        return []

# JSON에서 임베딩 벡터와 메타데이터 추출
def extract_vectors_and_metadata(embedded_data):
    vectors = []
    metadatas = []
    
    if not isinstance(embedded_data, list):
        st.error("임베딩 데이터가 리스트 형식이 아닙니다.")
        st.write("임베딩 데이터 구조 확인:", embedded_data)
        return [], []
    
    for idx, item in enumerate(embedded_data):
        if isinstance(item, dict):
            if all(key in item for key in ['임베딩', '제목', '요약', '세부인정사항']):
                try:
                    vectors.append(np.array(item['임베딩']))
                    metadatas.append({
                        "제목": item["제목"],
                        "요약": item["요약"],
                        "세부인정사항": item["세부인정사항"]
                    })
                except (TypeError, ValueError) as e:
                    st.warning(f"임베딩 데이터를 배열로 변환하는 중 오류 발생 (인덱스 {idx}): {e}")
                    st.write(f"문제가 있는 임베딩 데이터 내용: {item['임베딩']}")
                    continue
            else:
                st.warning(f"필수 키가 누락된 아이템 발견 (인덱스 {idx}): {item}")
        else:
            st.warning(f"비정상적인 데이터 형식의 아이템 발견 (인덱스 {idx}): {item}")
    
    return vectors, metadatas

# 사용자 입력을 구조화하는 함수
def structure_user_input(user_input):
    try:
        prompt_template = st.secrets["openai"]["prompt_structuring"]
        prompt = prompt_template.format(user_input=user_input)

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 의료 기록을 구조화하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5,
        )

        structured_input = response.choices[0].message.content.strip()
        return structured_input

    except Exception as e:
        st.error(f"입력 구조화 중 오류 발생: {e}")
        return None

# 사용자 입력을 임베딩하는 함수
def get_embedding_from_openai(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {e}")
        return None

# 코사인 유사도를 계산하여 상위 N개 결과 반환
def find_top_n_similar(embedding, vectors, metadatas, top_n=5):
    if len(vectors) != len(metadatas):
        st.error(f"벡터 수와 메타데이터 수가 일치하지 않습니다: 벡터 수 = {len(vectors)}, 메타데이터 수 = {len(metadatas)}")
        return []

    user_embedding = np.array(embedding).reshape(1, -1)
    similarities = cosine_similarity(user_embedding, vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    top_results = [{"유사도": similarities[i], "메타데이터": metadatas[i]} for i in top_indices]
    
    return top_results

# GPT-4 모델을 사용하여 연관성 점수를 평가하는 함수
def evaluate_relevance_with_gpt(structured_input, items):
    try:
        prompt_template = st.secrets["openai"]["prompt_scoring"]
        formatted_items = "\n\n".join([f"항목 {i+1}: {item['요약']}" for i, item in enumerate(items)])
        prompt = prompt_template.format(user_input=structured_input, items=formatted_items)

        with st.spinner("연관성 점수 평가 중..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7,
            )

        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        st.error(f"GPT 모델 호출 중 오류 발생: {e}")
        return None

# 점수 추출 함수
def extract_scores(full_response, num_items):
    scores = {}
    for idx in range(1, num_items + 1):
        score_match = re.search(rf"항목 {idx}:\s*(\d+)", full_response)
        if score_match:
            scores[idx] = int(score_match.group(1))
    return scores if len(scores) == num_items else None  # 모든 스코어를 추출하지 못하면 None 반환

# 재시도 케이스: 스코어 추출 실패 시 재시도
def retry_scoring_gpt(structured_input, items):
    if st.session_state.score_parsing_attempt < st.session_state.max_attempts:
        st.session_state.score_parsing_attempt += 1
        st.warning(f"스코어 추출에 실패했습니다. 스코어링 GPT를 다시 호출합니다... (시도 {st.session_state.score_parsing_attempt}/{st.session_state.max_attempts})")
        # 스코어링 GPT 다시 호출
        new_response = evaluate_relevance_with_gpt(structured_input, items)
        return new_response
    else:
        st.warning("스코어 추출에 여러 번 실패했습니다. '삭감 여부 확인' 버튼을 다시 눌러주세요.")
        return None

# 재시도 케이스: 연관된 항목이 없는 경우 임베딩 및 검색 재시도
def retry_embedding_and_search(department, user_input, vectors, metadatas):
    if st.session_state.embedding_search_attempt < st.session_state.max_attempts:
        st.session_state.embedding_search_attempt += 1
        st.warning(f"연관성이 떨어지는 결과가 나왔습니다. 임베딩 및 검색 과정을 다시 수행합니다... (시도 {st.session_state.embedding_search_attempt}/{st.session_state.max_attempts})")
        
        # 데이터가 이미 로드되어 있는지 확인
        if not vectors or not metadatas:
            # 데이터가 없다면 로드
            embedded_data = load_data_if_department_selected(department)
            if not embedded_data:
                st.error("데이터 로드 실패, 또는 해당 분과의 데이터가 아직 없습니다.")
                return False
            vectors, metadatas = extract_vectors_and_metadata(embedded_data)
            st.session_state.vectors = vectors
            st.session_state.metadatas = metadatas
        else:
            # 데이터가 이미 로드되어 있으므로 재로딩하지 않음
            vectors = st.session_state.vectors
            metadatas = st.session_state.metadatas
        
        # 사용자 입력 처리
        structured_input, embedding = process_user_input(user_input)
        if not structured_input or not embedding:
            return False
        
        st.session_state.structured_input = structured_input
        st.session_state.embedding = embedding

        # 검색된 급여기준 및 분석 결과 출력
        relevant_results, full_response = display_results(embedding, vectors, metadatas, structured_input)
        if relevant_results:
            # 개별 기준에 대한 분석
            overall_decision, explanations = analyze_criteria(relevant_results, user_input)
            st.session_state.overall_decision = overall_decision
            st.session_state.explanations = explanations
            st.session_state.relevant_results = relevant_results
            st.session_state.full_response = full_response
            st.session_state.results_displayed = True
            st.session_state.retry_type = None
            return True
        else:
            st.warning("재시도 후에도 연관성 높은 항목을 찾지 못했습니다.")
            return False
    else:
        st.warning("임베딩 및 검색 재시도 횟수를 초과했습니다.")
        return False

# 재시도 로직 처리 함수
def handle_retries(department, user_input):
    if st.session_state.retry_attempts >= st.session_state.max_attempts:
        st.warning("죄송합니다. 응답 과정에서 문제가 발생했습니다. 다시 시도하려면 '삭감 여부 확인' 버튼을 한 번 더 눌러주세요.")
        st.session_state.retry_type = None
        return

    st.session_state.retry_attempts += 1  # 재시도 횟수 증가

    if st.session_state.retry_type == 'score_parsing':
        new_response = retry_scoring_gpt(st.session_state.structured_input, st.session_state.metadatas)
        if new_response:
            scores = extract_scores(new_response, len(st.session_state.metadatas))
            if scores:
                st.session_state.full_response = new_response
                st.session_state.scores = scores

                relevant_results = []
                for idx, doc in enumerate(st.session_state.metadatas, 1):
                    score = scores.get(idx, None)
                    if score and score >= 7:
                        relevant_results.append(doc)
                if relevant_results:
                    # 개별 기준에 대한 분석
                    overall_decision, explanations = analyze_criteria(relevant_results, user_input)
                    st.session_state.overall_decision = overall_decision
                    st.session_state.explanations = explanations
                    st.session_state.relevant_results = relevant_results
                    st.session_state.results_displayed = True
                    st.session_state.retry_type = None
                    return
            # 스코어 추출 실패 또는 유의미한 결과 없음
            handle_retries(department, user_input)
        else:
            # GPT 호출 실패
            handle_retries(department, user_input)

    elif st.session_state.retry_type == 'embedding_search':
        retry_success = retry_embedding_and_search(department, user_input, st.session_state.vectors, st.session_state.metadatas)
        if retry_success:
            st.session_state.retry_type = None
            return
        else:
            handle_retries(department, user_input)


# 검색 결과 및 분석 결과를 출력하는 함수
def display_results(embedding, vectors, metadatas, structured_input):
    top_results = find_top_n_similar(embedding, vectors, metadatas)
    st.subheader("상위 유사 항목")
    for idx, result in enumerate(top_results, 1):
        with st.expander(f"항목 {idx} - {result['메타데이터']['제목']}"):
            st.write(f"제목: {result['메타데이터']['제목']}")
            st.write(f"요약: {result['메타데이터']['요약']}")

    items = [result['메타데이터'] for result in top_results]

    full_response = evaluate_relevance_with_gpt(structured_input, items)

    if full_response:
        scores = extract_scores(full_response, len(items))
        if not scores:
            st.warning("스코어 추출에 실패했습니다. 스코어링 GPT를 다시 호출합니다.")
            st.session_state.retry_type = 'score_parsing'
            return [], full_response
        else:
            st.session_state.full_response = full_response
            st.session_state.scores = scores

            st.subheader("연관성 평가 결과")
            with st.expander("연관성 평가 결과 상세 보기"):
                st.write(full_response)

            relevant_results = []
            for idx, doc in enumerate(items, 1):
                score = scores.get(idx, None)
                if score and score >= 7:
                    with st.expander(f"항목 {idx} (score: {score})"):
                        st.write(f"세부인정사항:")
                        st.write(doc['세부인정사항'])
                    relevant_results.append(doc)
            
            if not relevant_results:
                st.session_state.retry_type = 'embedding_search'
                return [], full_response
            else:
                return relevant_results, full_response
    else:
        st.error("죄송합니다. 일시적인 문제로 결과를 가져올 수 없습니다.")
        return None, None


def extract_text_between_numbers(structured_input):
    import re
    # 정규표현식을 사용하여 "2."와 "6." 사이의 텍스트를 추출
    pattern = r"수술 및 치료, 날짜나 기간\s*[:\-]?\s*(.*?)\s*치료재료"
    match = re.search(pattern, structured_input, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        return None


# 사용자 입력 처리 및 임베딩 생성
def process_user_input(user_input):
    try:
        with st.spinner("사용자 입력 분석중..."):
            structured_input = structure_user_input(user_input)
            if not structured_input:
                st.error("입력 텍스트 분석에 실패했습니다.")
                return None, None

        # st.success("입력 처리 완료")
        with st.expander("구조화된 입력 보기"):
            st.write(structured_input)

        # "2."부터 "6." 이전까지의 텍스트를 추출
        extracted_text = extract_text_between_numbers(structured_input)
        if not extracted_text:
            st.error("구조화된 입력에서 필요한 부분을 추출하지 못했습니다.")
            return None, None

        with st.spinner("임베딩 생성 중..."):
            embedding = get_embedding_from_openai(extracted_text)
            if not embedding:
                st.error("죄송합니다. 입력한 내용을 처리하는 중 문제가 발생했습니다.")
                return None, None
        
        # st.success("임베딩 생성 완료!")
        return structured_input, embedding
    except Exception as e:
        error_message = f"사용자 입력 처리 중 오류 발생: {e}"
        st.error(error_message)
        st.exception(e)
        st.session_state.setdefault('errors', []).append(error_message)
        return None, None

# 유효 기준에 대한 세부적인 분석과 심사
def analyze_criteria(relevant_results, user_input):
    explanations = []
    overall_decision = "삭감 안될 가능성 높음"

    prompt_template = st.secrets["openai"]["prompt_interpretation"]

    with st.spinner("개별 기준에 대한 심사 진행중..."):
        progress_bar = st.progress(0.0)
        total = len(relevant_results)
        for idx, criteria in enumerate(relevant_results, 1):
            try:
                prompt = prompt_template.format(user_input=user_input, criteria=criteria['세부인정사항'])
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 의료 문서를 분석하는 보험 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8192,  # 적절한 토큰 수로 조정
                    temperature=0.3,
                )
                analysis = response.choices[0].message.content.strip()

                index_of_4 = analysis.find("4.")
                if index_of_4 != -1:
                    content_after_4 = analysis[index_of_4+2:].strip()
                else:
                    content_after_4 = analysis

                explanations.append({
                    'index': idx,
                    'full_analysis': analysis,
                    'content_after_4': content_after_4
                })

                if "의료비는 삭감됩니다." in analysis:
                    overall_decision = "삭감될 가능성 높음"
                
                progress_bar.progress(idx / total)
            except Exception as e:
                st.error(f"기준 {idx}에 대한 분석 중 오류 발생: {e}")
                progress_bar.progress(idx / total)
    
    progress_bar.empty()
    return overall_decision, explanations

def display_results_and_analysis():
    if st.session_state.get('results_displayed', False):
        # 기존 판정 결과 표시
        st.subheader("심사 결과")
        st.write(st.session_state.overall_decision)

        # 개별 기준에 대한 심사 결과 표시
        st.subheader("개별 기준에 대한 심사 결과")
        for explanation in st.session_state.explanations:
            with st.expander(f"항목 {explanation['index']} - 상세 보기"):
                st.write(explanation['content_after_4'])

        # 업그레이드된 임상노트 생성 및 표시
        if st.session_state['upgraded_note'] is None:
            upgraded_note = generate_upgraded_clinical_note(
                st.session_state.overall_decision,
                st.session_state.user_input,
                st.session_state.explanations
            )
            st.session_state['upgraded_note'] = upgraded_note

        st.subheader("업그레이드된 임상노트")
        with st.expander("업그레이드된 임상노트 보기"):
            if 'upgraded_note' in st.session_state:
                upgraded_note = st.session_state['upgraded_note']
                note_area = st.text_area("업그레이드된 임상노트", value=upgraded_note, height=300)
                
            else:
                st.write("업그레이드된 임상노트를 생성하는 중 문제가 발생했습니다.")



# 채팅 기능 추가: 이전 내용들을 대화 내역에 추가하는 함수
def add_to_conversation(role, message):
    st.session_state.conversation.append({"role": role, "message": message})

# 대화 메시지를 표시하는 함수
def display_chat_messages():
    for chat in st.session_state.conversation:
        if chat['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(chat['message'])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat['message'])

# 채팅 인터페이스를 Sidebar에 표시하는 함수
def display_chat_interface():
    with st.sidebar:
        st.header("AI Assistant와 채팅을 시작해보세요.")
        display_chat_messages()

        # 사용자 입력받는 채팅 입력창
        if user_question := st.chat_input("질문을 입력하세요"):
            if user_question.strip() == "":
                st.warning("질문을 입력해주세요.")
            else:
                add_to_conversation('user', user_question)
                with st.chat_message("user"):
                    st.markdown(user_question)

                model_response = generate_chat_response(user_question)
                add_to_conversation('assistant', model_response)
                with st.chat_message("assistant"):
                    st.markdown(model_response)

# 채팅에서 응답을 생성하는 함수
def generate_chat_response(user_question):
    try:
        # 이전의 컨텍스트 가져오기 (최근 10개 메시지만)
        recent_conversation = st.session_state.conversation[-10:]
        conversation_history = ""
        for chat in recent_conversation:
            conversation_history += f"사용자: {chat['message']}\n" if chat['role'] == 'user' else f"모델: {chat['message']}\n"

        # explanations에서 최근 10개만 가져오기
        recent_explanations = st.session_state.explanations[-10:]
        explanations_texts = [explanation['full_analysis'] for explanation in recent_explanations]

        # GPT에게 전달할 프롬프트
        prompt_template = st.secrets["openai"]["prompt_chatting"]

        prompt = prompt_template.format(
            conversation_history=conversation_history,
            user_input=st.session_state.user_input,
            overall_decision=st.session_state.overall_decision,
            explanations='\n'.join(explanations_texts),
            user_question=user_question
        )
    
        with st.spinner("응답 생성 중..."):
            response = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "당신은 의료보험 분야의 전문가 어시스턴트입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )

        model_output = response.choices[0].message.content.strip()
        return model_output
    
    except Exception as e:
        st.error(f"응답 생성 중 오류 발생: {e}")
        st.exception(e)  # 예외의 전체 스택 트레이스 표시
        return "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다."


# 피드백 저장 함수
def save_feedback_to_s3():
    
    # 세션 상태에서 기존의 사용자 로그 데이터를 가져옴
    user_log_data = st.session_state.get('user_log_data', {})
    if not user_log_data:
        st.error("서비스를 이용하신 후 피드백을 보내주세요.")
        return
    
    # 피드백 추가
    user_log_data['feedback'] = st.session_state.get("feedback_text", "")
    
    # JSON 형식으로 변환
    feedback_json = json.dumps(user_log_data, ensure_ascii=False)
    
    # S3 버킷 정보 설정
    bucket_name = "medinsurance-assist-beta-user-log"
    file_name = f"user_logs/{user_log_data['timestamp']}_{user_log_data['session_id']}.json"
    
    # S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["access_key"],
        aws_secret_access_key=st.secrets["aws"]["secret_key"],
        region_name='ap-northeast-2'
    )
    
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=feedback_json)
        # st.success("피드백이 성공적으로 저장되었습니다.")
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")


# 피드백 입력창 추가
def feedback_section():
    with st.sidebar:
        # HTML과 CSS를 이용해 서브헤더 스타일 조정
        st.markdown("""
        <style>
        .feedback-header {
            font-size: 18px;
            color: #4CAF50;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # 작고 부담 없는 피드백 섹션
        st.markdown('<p class="feedback-header">개발자에게 피드백 보내기</p>', unsafe_allow_html=True)

        feedback_text = st.text_input("피드백을 입력해주세요", key="feedback_text")

        if st.button("피드백 전송!"):
            if feedback_text.strip() == "":
                st.warning("피드백을 입력해주세요.")
            else:
                save_feedback_to_s3()
                st.success("피드백이 전송되었습니다. 감사합니다!")


# 업그레이드된 임상노트를 생성하는 함수
def generate_upgraded_clinical_note(overall_decision, user_input, explanations):
    try:
        prompt_template = st.secrets["openai"]["prompt_upgrade_note"]

        # explanations에서 필요한 내용을 추출하여 explanations_text 생성
        explanations_text = "\n\n".join([
            f"\n{explanation.get('content_after_4', '')}"
            for explanation in explanations
        ])

        # 삭감 사유 또는 추가 개선 사항을 추출
        if overall_decision == "삭감될 가능성 높음":
            # 삭감 사유를 explanations에서 추출
            guidance = f"다음 삭감 사유를 고려하여 임상노트를 개선하세요:\n{explanations_text}"        
        else:
            # 추가적인 개선 사항 제안
            guidance = "임상노트를 더욱 완벽하게 만들기 위해 추가할 수 있는 내용을 추가하세요. 보존적 치료가 앞서 시행되었음을 강조하세요."

        prompt = prompt_template.format(
            overall_decision=overall_decision,
            guidance=guidance,
            user_input=user_input,
            explanations_text=explanations_text
        )

        with st.spinner("업그레이드된 임상노트 생성 중..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 의료 문서를 작성하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5,
            )

        upgraded_note = response.choices[0].message.content.strip()
        return upgraded_note

    except Exception as e:
        st.error(f"업그레이드된 임상노트 생성 중 오류 발생: {e}")
        st.exception(e)
        return None


# 메인 함수
def main():
    add_logo()
    st.title("의료비 삭감 판정 어시스트 - <삭감노노>")

    # 1. 사용자 정보 및 입력 수집
    occupation, other_occupation, department, user_input = collect_user_input()

    # 2. '삭감 여부 확인' 버튼 클릭 시 초기화 및 재시도 시작
    if st.button("삭감 여부 확인", disabled=st.session_state.get('button_disabled', True)):
        if not st.session_state.get('agree_to_collect', False):
            st.warning("사용자 정보 수집에 동의해야 합니다.")
        elif not st.session_state.is_clinical_note:
            st.warning("유효한 임상노트를 입력해주세요.")
        elif not department:
            st.warning("분과를 선택해주세요.")
        else:
            save_user_log_to_s3()
            st.session_state.conversation = []
            st.session_state.results_displayed = False
            st.session_state.score_parsing_attempt = 0
            st.session_state.embedding_search_attempt = 0
            st.session_state.overall_decision = ""
            st.session_state.explanations = []
            st.session_state.relevant_results = []
            st.session_state.full_response = ""
            st.session_state.scores = {}
            st.session_state.retry_type = None  # 재시도 유형 초기화
            st.session_state.retry_attempts = 0
            st.session_state.upgraded_note = None
            st.session_state.copy_text = ''

            st.session_state.structured_input, st.session_state.embedding = process_user_input(user_input)
            if not st.session_state.structured_input or not st.session_state.embedding:
                return

            # 임베딩 데이터 로드
            if not st.session_state.vectors or not st.session_state.metadatas:
                embedded_data = load_data_if_department_selected(department)
                if not embedded_data:
                    st.error("데이터 로드 실패, 또는 해당 분과의 데이터가 아직 없습니다.")
                    return

                st.session_state.vectors, st.session_state.metadatas = extract_vectors_and_metadata(embedded_data)
                if not st.session_state.vectors:
                    st.error("임베딩 데이터 추출에 실패했습니다.")
                    return

            # 검색된 급여기준 및 분석 결과 출력
            relevant_results, full_response = display_results(
                st.session_state.embedding,
                st.session_state.vectors,
                st.session_state.metadatas,
                st.session_state.structured_input
            )

            if not relevant_results:
                st.session_state.retry_type = 'embedding_search'
                handle_retries(department, user_input)
            else:
                # 개별 기준에 대한 분석
                overall_decision, explanations = analyze_criteria(relevant_results, user_input)
                st.session_state.overall_decision = overall_decision
                st.session_state.explanations = explanations
                st.session_state.relevant_results = relevant_results
                st.session_state.full_response = full_response
                st.session_state.results_displayed = True

    # 3. 재시도 처리
    if st.session_state.retry_type:
        handle_retries(department, user_input)

    # 4. 결과 표시
    display_results_and_analysis()

    # Sidebar에 채팅 인터페이스 표시
    display_chat_interface()

    feedback_section()

if __name__ == "__main__":
    main()
