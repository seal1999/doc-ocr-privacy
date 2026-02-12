from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """\
당신은 문서 이미지에서 개인정보를 찾아 마스킹하는 전문가입니다.

반드시 다음 순서로 도구를 호출하세요:
1. ocr_image: 이미지 경로를 입력받아 텍스트를 추출합니다.
2. detect_pii: 추출된 텍스트에서 개인정보(PII)를 검출합니다.
3. mask_pii: 검출된 PII를 마스킹합니다. text에는 ocr_image의 full_text를, \
pii_entities_json에는 detect_pii의 entities 배열을 JSON 문자열로 전달하세요.

마지막으로 마스킹된 결과를 사용자에게 보여주세요.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


async def run(image_path: str, model: str = "llama3.2:latest") -> str:
    """이미지에서 OCR → PII 검출 → 마스킹 파이프라인을 실행합니다."""
    llm = ChatOllama(model=model, temperature=0)

    async with MultiServerMCPClient(_build_server_config()) as client:
        tools = client.get_tools()
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"다음 이미지를 처리해주세요: {image_path}"}]}
        )

        final_message = result["messages"][-1].content
        return final_message
