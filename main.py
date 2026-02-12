import argparse
import asyncio
import sys

from agent.ocr_privacy_agent import run


def main():
    parser = argparse.ArgumentParser(description="이미지에서 개인정보를 검출하고 마스킹합니다.")
    parser.add_argument("image_path", help="OCR을 수행할 이미지 파일 경로")
    parser.add_argument("--model", default="llama3.2:latest", help="Ollama 모델 이름 (기본값: llama3.2:latest)")
    args = parser.parse_args()

    result = asyncio.run(run(args.image_path, args.model))
    print(result)


if __name__ == "__main__":
    main()
