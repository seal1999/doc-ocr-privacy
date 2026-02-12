import json
import os

import easyocr
from fastmcp import FastMCP

mcp = FastMCP("ocr-server")

reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    global reader
    if reader is None:
        reader = easyocr.Reader(["ko", "en"], gpu=False)
    return reader


@mcp.tool()
def ocr_image(image_path: str) -> str:
    """이미지 파일에서 텍스트를 추출합니다 (한국어/영어 지원).

    Args:
        image_path: OCR을 수행할 이미지 파일의 절대 경로

    Returns:
        JSON 문자열: full_text(전체 텍스트)와 details(개별 인식 결과 목록)
    """
    if not os.path.exists(image_path):
        return json.dumps({"error": f"파일을 찾을 수 없습니다: {image_path}"}, ensure_ascii=False)

    r = get_reader()
    results = r.readtext(image_path)

    details = []
    text_parts = []
    for bbox, text, confidence in results:
        text_parts.append(text)
        details.append({
            "text": text,
            "confidence": round(confidence, 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    full_text = " ".join(text_parts)
    return json.dumps({"full_text": full_text, "details": details}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
