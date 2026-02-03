# VibeVoice ASR 마이크 테스트 가이드

VibeVoice-ASR 모델을 사용하여 실시간으로 마이크 입력을 받아 텍스트로 변환하는 테스트 환경입니다.

## 1. 시스템 요구사항

### 필수
- Python 3.8 이상
- 마이크 (내장 또는 외장)
- 디스크 공간: 20GB 이상 (모델 다운로드용)

### 권장 (GPU 사용 시)
- NVIDIA GPU (VRAM 8GB 이상)
- CUDA 11.8 이상
- cuDNN 8.x

> **참고**: GPU가 없어도 CPU 모드로 실행 가능합니다 (속도가 느림).

## 2. 의존성 설치

```bash
cd VibeVoice

# 필수 라이브러리
pip install sounddevice soundfile

# GPU 사용 시 (선택)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 3. CLI 기반 마이크 테스트 (추천)

### 간편 실행 (Windows)

`run_mic_test.bat` 파일을 더블 클릭하거나:

```bash
cd c:\Repos\Call_Center_Voice\VibeVoice
.\run_mic_test.bat
```

### Python 직접 실행

```bash
# 기본 실행 (5초 녹음)
python test_mic_cli.py

# 녹음 시간 변경
python test_mic_cli.py --duration 10

# 마이크 장치 확인
python test_mic_cli.py --list-devices

# 특정 마이크 사용
python test_mic_cli.py --mic 1

# CPU 모드 강제
python test_mic_cli.py --device cpu

# 녹음 파일 유지
python test_mic_cli.py --keep-audio
```

### CLI 옵션 전체

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--duration` | 녹음 시간 (초) | 5 |
| `--device` | 추론 장치 (auto/cuda/cpu) | auto |
| `--mic` | 마이크 장치 인덱스 | 기본 마이크 |
| `--model_path` | 모델 경로/ID | microsoft/VibeVoice-ASR |
| `--list-devices` | 오디오 장치 목록 표시 | - |
| `--keep-audio` | 녹음 파일 삭제 안 함 | - |

## 4. Web UI (Gradio) 테스트

웹 인터페이스에서 마이크 테스트:

```bash
cd c:\Repos\Call_Center_Voice\VibeVoice
python demo/vibevoice_asr_gradio_demo.py
```

브라우저에서 `http://127.0.0.1:7860` 접속 후:
1. Audio 입력에서 "Microphone" 선택
2. 녹음 버튼 클릭
3. "Transcribe" 버튼으로 전사

### LLM 분석 기능 활성화 (선택)

통화 분석 AI 기능을 사용하려면:

#### GLM-4 사용 (기본, 무료 티어 지원)

```bash
# 환경변수 설정
set ZHIPUAI_API_KEY=your-glm-api-key

# 또는 .env 파일 생성
echo ZHIPUAI_API_KEY=your-glm-api-key > .env
```

#### OpenAI 사용

```bash
# 환경변수 설정
set OPENAI_API_KEY=your-openai-api-key

# llm_client.py에서 provider 변경 필요:
# LLMClient(provider="openai")
```

## 5. 파일 구성

| 파일 | 설명 |
|------|------|
| `test_mic_cli.py` | CLI 마이크 테스트 스크립트 |
| `run_mic_test.bat` | Windows 실행 배치 파일 |
| `demo/vibevoice_asr_gradio_demo.py` | Gradio 웹 데모 |
| `vibevoice/llm_client.py` | LLM 분석 클라이언트 |

## 6. 트러블슈팅

### 마이크를 찾을 수 없음

```bash
# 장치 목록 확인
python test_mic_cli.py --list-devices

# 특정 마이크 지정
python test_mic_cli.py --mic <장치번호>
```

### CUDA 메모리 부족

```bash
# CPU 모드로 실행
python test_mic_cli.py --device cpu
```

### 모델 다운로드 실패

1. 인터넷 연결 확인
2. 디스크 공간 확인 (약 18GB 필요)
3. HuggingFace 캐시 삭제 후 재시도:
   ```bash
   # Windows
   rmdir /s %USERPROFILE%\.cache\huggingface\hub\models--microsoft--VibeVoice-ASR
   ```

### sounddevice 설치 오류 (Windows)

```bash
# Microsoft Visual C++ Build Tools 설치 필요할 수 있음
pip install sounddevice --upgrade
```

### 녹음이 안 됨

1. Windows 설정 → 개인 정보 → 마이크 → 앱 접근 허용
2. 다른 앱이 마이크 사용 중인지 확인
3. 마이크 음소거 해제 확인

## 7. 첫 실행 시 참고사항

- **모델 다운로드**: 첫 실행 시 약 18GB 모델을 다운로드합니다 (네트워크에 따라 10~30분 소요)
- **GPU 초기화**: CUDA 사용 시 첫 추론에 몇 초 추가 소요
- **캐시**: 이후 실행은 캐시된 모델을 사용하여 빠르게 시작됩니다

---

**구현 노트:**
- `sounddevice` 라이브러리로 마이크 입력 캡처
- GPU 자동 감지 (없으면 CPU 폴백)
- 실시간 녹음 진행률 표시
- OpenAI API 연동 LLM 분석 지원
