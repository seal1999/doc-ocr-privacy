import json
import re

from fastmcp import FastMCP

mcp = FastMCP("pii-detection-server")

PII_PATTERNS: dict[str, re.Pattern] = {
    "주민등록번호": re.compile(r"\d{6}\s*[-–]\s*[1-4]\d{6}"),
    "휴대폰번호": re.compile(r"01[016789]\s*[-–.]?\s*\d{3,4}\s*[-–.]?\s*\d{4}"),
    "유선전화번호": re.compile(r"0[2-6][0-5]?\s*[-–.]?\s*\d{3,4}\s*[-–.]?\s*\d{4}"),
    "이메일": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "카드번호": re.compile(r"\d{4}\s*[-–.]?\s*\d{4}\s*[-–.]?\s*\d{4}\s*[-–.]?\s*\d{4}"),
    "계좌번호": re.compile(r"\d{3,6}\s*[-–]\s*\d{2,6}\s*[-–]\s*\d{1,6}(?:\s*[-–]\s*\d{1,6})?"),
    "운전면허번호": re.compile(r"\d{2}\s*[-–]\s*\d{2}\s*[-–]\s*\d{6}\s*[-–]\s*\d{2}"),
    "여권번호": re.compile(r"[A-Z]{1,2}\d{7,8}"),
}


@mcp.tool()
def detect_pii(text: str) -> str:
    """텍스트에서 개인정보(PII)를 검출합니다.

    Args:
        text: PII를 검출할 텍스트

    Returns:
        JSON 문자열: pii_count(검출 건수)와 entities(검출된 PII 목록)
    """
    entities = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

    entities.sort(key=lambda e: e["start"])
    return json.dumps({"pii_count": len(entities), "entities": entities}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
