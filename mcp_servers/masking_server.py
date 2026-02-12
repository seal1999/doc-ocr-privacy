import json

from fastmcp import FastMCP

mcp = FastMCP("masking-server")

MASK_MAP: dict[str, str] = {
    "주민등록번호": "******-*******",
    "휴대폰번호": "***-****-****",
    "유선전화번호": "***-****-****",
    "이메일": "****@****",
    "카드번호": "****-****-****-****",
    "계좌번호": "***-***-******",
    "운전면허번호": "**-**-******-**",
    "여권번호": "***********",
}


@mcp.tool()
def mask_pii(text: str, pii_entities_json: str | list) -> str:
    """검출된 PII를 마스킹 처리합니다.

    Args:
        text: 원본 텍스트
        pii_entities_json: detect_pii에서 반환된 entities 배열 (JSON 문자열 또는 리스트)
            예: [{"type": "휴대폰번호", "value": "010-1234-5678", "start": 0, "end": 13}]

    Returns:
        JSON 문자열: masked_text(마스킹된 텍스트), masked_count(마스킹 건수), details(상세 내역)
    """
    if isinstance(pii_entities_json, list):
        entities = pii_entities_json
    else:
        try:
            entities = json.loads(pii_entities_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "pii_entities_json 파싱 실패"}, ensure_ascii=False)

    # 뒤에서부터 치환하여 인덱스 보존
    entities_sorted = sorted(entities, key=lambda e: e["start"], reverse=True)

    masked_text = text
    details = []
    for entity in entities_sorted:
        pii_type = entity["type"]
        start = entity["start"]
        end = entity["end"]
        original = entity["value"]
        mask = MASK_MAP.get(pii_type, "*" * len(original))

        masked_text = masked_text[:start] + mask + masked_text[end:]
        details.append({
            "type": pii_type,
            "original": original,
            "masked": mask,
        })

    details.reverse()
    return json.dumps(
        {"masked_text": masked_text, "masked_count": len(details), "details": details},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run()
