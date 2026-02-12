import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, create_react_agent

SYSTEM_PROMPT = """\
당신은 문서 이미지에서 개인정보를 찾아 마스킹하는 전문가입니다.

사용자가 이미지 경로를 제공하면, 반드시 아래 3개의 도구를 순서대로 모두 호출해야 합니다.
도구를 하나씩 호출하고, 각 도구의 결과를 다음 도구의 입력으로 사용하세요.
절대로 도구 호출을 생략하지 마세요. 3개 모두 호출해야 합니다.

## 도구 호출 순서

### Step 1: ocr_image
- image_path: 사용자가 제공한 이미지 파일 경로
- 결과: JSON에서 "full_text" 값을 기억하세요 (이후 Step 2, 3에서 사용)

### Step 2: detect_pii
- text: Step 1에서 얻은 full_text 값을 그대로 전달
- 결과: JSON에서 "entities" 배열을 기억하세요 (Step 3에서 사용)

### Step 3: mask_pii (반드시 호출할 것!)
- text: Step 1에서 얻은 full_text 값을 그대로 전달
- pii_entities_json: Step 2에서 얻은 entities 배열을 JSON 문자열로 전달
- 결과: "masked_text" 값이 최종 결과입니다

## 중요
- 3개의 도구를 모두 호출한 후에만 최종 응답을 작성하세요.
- mask_pii를 호출하지 않으면 작업이 완료되지 않은 것입니다.
- 최종 응답에서는 mask_pii의 masked_text 결과를 보여주세요.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("ocr_privacy_agent")


def _setup_logger(log_path: Path) -> None:
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)


def _build_server_config() -> dict:
    uv = "uv"
    return {
        "ocr-server": {
            "command": uv,
            "args": ["run", "python", str(PROJECT_ROOT / "mcp_servers" / "ocr_server.py")],
            "transport": "stdio",
        },
        "pii-detection-server": {
            "command": uv,
            "args": ["run", "python", str(PROJECT_ROOT / "mcp_servers" / "pii_detection_server.py")],
            "transport": "stdio",
        },
        "masking-server": {
            "command": uv,
            "args": ["run", "python", str(PROJECT_ROOT / "mcp_servers" / "masking_server.py")],
            "transport": "stdio",
        },
    }


def _extract_tool_text(content) -> str:
    """도구 메시지의 content에서 텍스트를 추출합니다.

    content가 리스트([{'type': 'text', 'text': '...'}]) 또는 문자열일 수 있습니다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item["text"]
    return str(content)


async def run(image_path: str, model: str = "llama3.2:latest") -> str:
    """이미지에서 OCR → PII 검출 → 마스킹을 ReAct Agent로 실행합니다."""
    log_path = Path(image_path).with_suffix(".log")
    _setup_logger(log_path)

    logger.info("에이전트 시작: image_path=%s, model=%s", image_path, model)

    llm = ChatOllama(model=model, temperature=0)

    client = MultiServerMCPClient(_build_server_config())
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}
    logger.info("MCP 도구 로드 완료: %s", list(tool_map.keys()))

    tool_node = ToolNode(tools, handle_tool_errors=True)
    agent = create_react_agent(model=llm, tools=tool_node, prompt=SYSTEM_PROMPT)

    logger.info("에이전트 실행 시작")
    messages = []
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=f"이 이미지에서 개인정보를 찾아 마스킹해주세요: {image_path}")]},
            config={"recursion_limit": 10},
        )
        messages = result["messages"]
    except Exception as e:
        logger.warning("에이전트 실행 중 예외 발생: %s", e)

    # 메시지 히스토리 로깅
    for msg in messages:
        logger.info("[%s] %s", msg.type, _extract_tool_text(msg.content)[:500])

    # mask_pii 결과에서 masked_text 추출 시도
    masked_text = _find_masked_text(messages)

    # 폴백: Agent가 mask_pii를 호출하지 못한 경우
    if masked_text is None:
        logger.warning("Agent가 mask_pii를 호출하지 못함 - 폴백 실행")
        masked_text = await _fallback_masking(messages, tool_map, image_path)

    if masked_text is None:
        logger.error("마스킹 결과를 얻지 못함")
        masked_text = "마스킹 결과를 생성하지 못했습니다."

    logger.info("최종 결과: %s", masked_text)
    return masked_text


def _find_masked_text(messages) -> str | None:
    """메시지 히스토리에서 mask_pii 도구 결과의 masked_text를 추출합니다."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "mask_pii" and msg.status != "error":
            try:
                text = _extract_tool_text(msg.content)
                data = json.loads(text)
                return data.get("masked_text")
            except (json.JSONDecodeError, AttributeError):
                continue
    return None


async def _fallback_masking(messages, tool_map: dict, image_path: str) -> str | None:
    """Agent가 실패한 경우, OCR → detect_pii → mask_pii를 직접 순차 호출합니다."""
    # 메시지에서 OCR 결과 추출 시도
    ocr_full_text = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        text = _extract_tool_text(msg.content)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            continue
        if msg.name == "ocr_image" and "full_text" in data:
            ocr_full_text = data["full_text"]

    # OCR 결과가 없으면 직접 호출
    if ocr_full_text is None:
        logger.info("폴백: ocr_image 직접 호출")
        ocr_result_raw = await tool_map["ocr_image"].ainvoke({"image_path": image_path})
        ocr_text = _extract_tool_text(ocr_result_raw)
        ocr_data = json.loads(ocr_text)
        ocr_full_text = ocr_data["full_text"]

    # detect_pii 직접 호출
    logger.info("폴백: detect_pii 직접 호출")
    pii_result_raw = await tool_map["detect_pii"].ainvoke({"text": ocr_full_text})
    pii_text = _extract_tool_text(pii_result_raw)
    pii_data = json.loads(pii_text)
    logger.info("폴백: PII 검출 %d건", pii_data["pii_count"])

    # mask_pii 직접 호출
    logger.info("폴백: mask_pii 직접 호출")
    entities_json = json.dumps(pii_data["entities"], ensure_ascii=False)
    mask_result_raw = await tool_map["mask_pii"].ainvoke({
        "text": ocr_full_text,
        "pii_entities_json": entities_json,
    })
    mask_text = _extract_tool_text(mask_result_raw)
    mask_data = json.loads(mask_text)
    return mask_data.get("masked_text")
