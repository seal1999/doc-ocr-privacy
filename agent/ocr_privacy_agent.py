import json
import logging
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """\
당신은 문서 이미지에서 개인정보를 찾아 마스킹하는 전문가입니다.

반드시 다음 순서대로 도구를 하나씩 호출하세요. 한 번에 하나의 도구만 호출하세요:

Step 1: ocr_image 도구를 호출하여 이미지에서 텍스트를 추출합니다.
Step 2: detect_pii 도구를 호출하여 추출된 텍스트에서 개인정보(PII)를 검출합니다. \
text 파라미터에는 Step 1 결과의 full_text 값을 전달하세요.
Step 3: mask_pii 도구를 호출하여 PII를 마스킹합니다. \
text 파라미터에는 Step 1 결과의 full_text 값을, \
pii_entities_json 파라미터에는 Step 2 결과의 entities 배열을 JSON 문자열로 전달하세요.

모든 도구 호출이 끝나면 mask_pii의 masked_text 결과를 사용자에게 보여주세요.
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
    """이미지에서 OCR → PII 검출 → 마스킹 파이프라인을 실행합니다."""
    log_path = Path(image_path).with_suffix(".log")
    _setup_logger(log_path)

    logger.info("파이프라인 시작: image_path=%s, model=%s", image_path, model)

    llm = ChatOllama(model=model, temperature=0)

    client = MultiServerMCPClient(_build_server_config())
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}
    logger.info("MCP 도구 로드 완료: %s", list(tool_map.keys()))

    # Step 1: OCR
    logger.info("Step 1: ocr_image 호출")
    ocr_result_raw = await tool_map["ocr_image"].ainvoke({"image_path": image_path})
    ocr_text = _extract_tool_text(ocr_result_raw)
    ocr_data = json.loads(ocr_text)
    full_text = ocr_data["full_text"]
    logger.info("OCR 결과: %s", full_text)

    # Step 2: PII 검출
    logger.info("Step 2: detect_pii 호출")
    pii_result_raw = await tool_map["detect_pii"].ainvoke({"text": full_text})
    pii_text = _extract_tool_text(pii_result_raw)
    pii_data = json.loads(pii_text)
    logger.info("PII 검출 결과: %d건 - %s", pii_data["pii_count"], pii_data["entities"])

    # Step 3: 마스킹
    logger.info("Step 3: mask_pii 호출")
    entities_json = json.dumps(pii_data["entities"], ensure_ascii=False)
    mask_result_raw = await tool_map["mask_pii"].ainvoke({
        "text": full_text,
        "pii_entities_json": entities_json,
    })
    mask_text = _extract_tool_text(mask_result_raw)
    mask_data = json.loads(mask_text)
    masked_text = mask_data["masked_text"]
    logger.info("마스킹 결과: %s", masked_text)

    # LLM 요약
    logger.info("LLM 요약 생성")
    summary = await llm.ainvoke(
        f"다음은 이미지에서 추출한 텍스트의 개인정보를 마스킹한 결과입니다. "
        f"원본에서 {pii_data['pii_count']}건의 개인정보가 검출되어 마스킹되었습니다.\n\n"
        f"마스킹된 텍스트:\n{masked_text}\n\n"
        f"위 결과를 사용자에게 보기 좋게 정리해서 보여주세요."
    )

    logger.info("파이프라인 완료")
    return masked_text
