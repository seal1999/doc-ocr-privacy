# doc-ocr-privacy

이미지를 입력받아 **OCR 수행 → 개인정보(PII) 패턴 검출 → 마스킹된 결과 출력**까지를 LangChain 기반 Single Agent가 MCP 도구를 활용하여 자동 처리하는 프로젝트입니다.

## 아키텍처

```
main.py (CLI)
    │
    ▼
LangChain ReAct Agent (ChatOllama)
    │
    ├── MCP Server #1 (stdio) → ocr_image()     [EasyOCR]
    ├── MCP Server #2 (stdio) → detect_pii()    [Regex]
    └── MCP Server #3 (stdio) → mask_pii()      [String Replace]
```

## 프로젝트 구조

```
doc-ocr-privacy/
├── pyproject.toml                   # 의존성 정의
├── .python-version                  # Python 3.12
├── .gitignore
├── main.py                          # CLI 진입점
├── mcp_servers/
│   ├── ocr_server.py                # MCP Server #1: EasyOCR (한국어/영어)
│   ├── pii_detection_server.py      # MCP Server #2: 정규식 PII 검출 (8종)
│   └── masking_server.py            # MCP Server #3: PII 마스킹
├── agent/
│   └── ocr_privacy_agent.py         # LangChain ReAct Agent
└── samples/                         # 테스트용 이미지
```

## 의존성

| 패키지 | 용도 |
|--------|------|
| `easyocr` | OCR 엔진 (한국어/영어 지원) |
| `fastmcp` | MCP 서버 프레임워크 |
| `langchain` | 에이전트 프레임워크 |
| `langchain-ollama` | Ollama LLM 연동 |
| `langchain-mcp-adapters` | MCP ↔ LangChain 브릿지 |
| `langgraph` | 에이전트 실행 그래프 |

## MCP 서버 상세

### Server #1 - OCR (`ocr_server.py`)
- **도구**: `ocr_image(image_path)`
- EasyOCR Reader(`['ko', 'en']`)로 이미지에서 텍스트 추출
- 반환: `full_text`, `details` (텍스트, 신뢰도, bbox)

### Server #2 - PII 검출 (`pii_detection_server.py`)
- **도구**: `detect_pii(text)`
- 한국어 PII 정규식 패턴 8종 지원:
  - 주민등록번호, 휴대폰번호, 유선전화번호, 이메일
  - 카드번호, 계좌번호, 운전면허번호, 여권번호
- 반환: `pii_count`, `entities` (유형, 값, 위치)

### Server #3 - 마스킹 (`masking_server.py`)
- **도구**: `mask_pii(text, pii_entities_json)`
- PII 유형별 마스크 패턴으로 치환 (뒤→앞 순서로 인덱스 보존)
- 반환: `masked_text`, `masked_count`, `details`

## 사용 방법

### 1. 환경 설치

```bash
uv sync
```

### 2. Ollama 모델 준비

```bash
ollama pull llama3.2
```

### 3. 실행

```bash
# 기본 실행 (llama3.2)
uv run python main.py samples/sample_image.png

# 다른 모델로 실행 (모델 설치 후 사용)
ollama pull qwen2.5:7b
uv run python main.py samples/sample_image.png --model qwen2.5:7b
```

## 참고 사항

- **Python 3.12**: PyTorch 호환성을 위해 `.python-version`으로 고정
- **EasyOCR 첫 실행**: 모델 파일 다운로드 (~100MB)가 필요하며, 이후에는 캐시 사용
- **llama3.2 (3B) tool calling**: 시스템 프롬프트에 도구 호출 순서를 명시하고 `temperature=0`으로 안정성 확보. 문제 발생 시 `--model` 플래그로 더 큰 모델 사용 가능
